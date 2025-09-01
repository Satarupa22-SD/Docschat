import os
from pathlib import Path
from typing import List, Dict
import re
from dataclasses import dataclass

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    import PyPDF2
except ImportError:
    chromadb = None
    SentenceTransformer = None
    PyPDF2 = None

@dataclass
class Reference:
    line_number: int
    page_number: int
    content: str
    section: str
    similarity_score: float

class ResearchDatabase:
    def __init__(self, db_path: str = "./research_db"):
        if chromadb is None or SentenceTransformer is None or PyPDF2 is None:
            raise ImportError("Missing required libraries: chromadb, sentence-transformers, PyPDF2")
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="research_papers",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def add_paper(self, paper_path: str, paper_id: str = None) -> bool:
        try:
            if paper_id is None:
                paper_id = Path(paper_path).stem
            text_chunks = self._extract_text_from_pdf(paper_path)
            try:
                self.collection.delete(where={"paper_id": paper_id})
            except:
                pass
            documents = []
            metadatas = []
            ids = []
            for i, chunk in enumerate(text_chunks):
                chunk_id = f"{paper_id}_chunk_{i}"
                documents.append(chunk['content'])
                metadatas.append({
                    "paper_id": paper_id,
                    "paper_path": paper_path,
                    "line_number": chunk['line_number'],
                    "page_number": chunk['page_number'],
                    "section": chunk['section']
                })
                ids.append(chunk_id)
            embeddings = self.embedding_model.encode(documents).tolist()
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            return True
        except Exception as e:
            print(f"Error adding paper: {e}")
            return False

    def _extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        chunks = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if not text:
                    continue
                lines = text.split('\n')
                current_chunk = []
                current_line = 1
                for line in lines:
                    line = line.strip()
                    if line:
                        current_chunk.append(line)
                        current_line += 1
                        if len(' '.join(current_chunk)) > 200 or self._is_section_break(line):
                            if current_chunk:
                                chunks.append({
                                    'content': ' '.join(current_chunk),
                                    'line_number': current_line - len(current_chunk),
                                    'page_number': page_num,
                                    'section': self._detect_section(current_chunk[0])
                                })
                            current_chunk = []
                if current_chunk:
                    chunks.append({
                        'content': ' '.join(current_chunk),
                        'line_number': current_line - len(current_chunk),
                        'page_number': page_num,
                        'section': self._detect_section(current_chunk[0])
                    })
        return chunks

    def _is_section_break(self, line: str) -> bool:
        section_patterns = [
            r'^\d+\.\s+[A-Z]',
            r'^[A-Z\s]+$',
            r'^References?$',
            r'^Bibliography$',
        ]
        for pattern in section_patterns:
            if re.match(pattern, line):
                return True
        return False

    def _detect_section(self, line: str) -> str:
        common_sections = [
            'abstract', 'introduction', 'methodology', 'results',
            'discussion', 'conclusion', 'references', 'background',
            'literature review', 'experiment', 'analysis'
        ]
        line_lower = line.lower()
        for section in common_sections:
            if section in line_lower:
                return section.title()
        return 'Content'

    def query(self, query: str, n_results: int = 5) -> List[Reference]:
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            references = []
            for i in range(len(results['documents'][0])):
                ref = Reference(
                    line_number=results['metadatas'][0][i]['line_number'],
                    page_number=results['metadatas'][0][i]['page_number'],
                    content=results['documents'][0][i],
                    section=results['metadatas'][0][i]['section'],
                    similarity_score=1 - results['distances'][0][i]
                )
                references.append(ref)
            return references
        except Exception as e:
            print(f"Error querying database: {e}")
            return []

    def query_all_papers(self, query: str, n_results: int = 5) -> List[Reference]:
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results * 20  # get more results to ensure paper diversity
            )
            references = []
            for i in range(len(results['documents'][0])):
                meta = results['metadatas'][0][i]
                ref = Reference(
                    line_number=meta['line_number'],
                    page_number=meta['page_number'],
                    content=results['documents'][0][i],
                    section=meta['section'],
                    similarity_score=1 - results['distances'][0][i]
                )
                # Attach paper_path for multi-paper context
                setattr(ref, 'paper_path', meta.get('paper_path', ''))
                references.append(ref)
            
            # Ensure diversity across papers by selecting top results from different papers
            paper_groups = {}
            for ref in references:
                paper_path = getattr(ref, 'paper_path', 'unknown')
                if paper_path not in paper_groups:
                    paper_groups[paper_path] = []
                paper_groups[paper_path].append(ref)
            
            # Sort each paper's results by similarity score
            for paper_path in paper_groups:
                paper_groups[paper_path].sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Select top results from each paper, ensuring diversity
            diverse_references = []
            papers_used = set()
            
            # First, get the top result from each paper
            for paper_path, refs in paper_groups.items():
                if refs and len(diverse_references) < n_results:
                    diverse_references.append(refs[0])
                    papers_used.add(paper_path)
            
            # Then fill remaining slots with best remaining results
            remaining_refs = []
            for paper_path, refs in paper_groups.items():
                remaining_refs.extend(refs[1:])  # Skip the first one we already used
            
            # Sort remaining by similarity and add best ones
            remaining_refs.sort(key=lambda x: x.similarity_score, reverse=True)
            for ref in remaining_refs:
                if len(diverse_references) < n_results:
                    diverse_references.append(ref)
                else:
                    break
            
            return diverse_references
        except Exception as e:
            print(f"Error querying all papers: {e}")
            return []

    def get_all_references_for_all_papers(self) -> List[Reference]:
        try:
            results = self.collection.get()
            references = []
            docs = results.get('documents', [])
            metas = results.get('metadatas', [])
            if docs and metas:
                for doc, meta in zip(docs, metas):
                    ref = Reference(
                        line_number=meta['line_number'],
                        page_number=meta['page_number'],
                        content=doc,
                        section=meta['section'],
                        similarity_score=1.0
                    )
                    setattr(ref, 'paper_path', meta.get('paper_path', ''))
                    references.append(ref)
            return references
        except Exception as e:
            print(f"Error retrieving all references: {e}")
            return []

    def get_all_references_for_paper(self, paper_path: str) -> List[Reference]:
        """
        Retrieve all text chunks for a given paper as Reference objects.
        """
        from pathlib import Path
        paper_id = Path(paper_path).stem
        try:
            results = self.collection.get(where={"paper_id": paper_id})
            references = []
            docs = results.get('documents', [])
            metas = results.get('metadatas', [])
            if docs and metas:
                for doc, meta in zip(docs, metas):
                    references.append(Reference(
                        line_number=meta['line_number'],
                        page_number=meta['page_number'],
                        content=doc,
                        section=meta['section'],
                        similarity_score=1.0  # Not relevant for summarization
                    ))
            return references
        except Exception as e:
            print(f"Error retrieving references for paper: {e}")
            return []

    def delete_paper(self, paper_path: str) -> bool:
        """
        Delete all chunks for a given paper from the database.
        """
        from pathlib import Path
        paper_id = Path(paper_path).stem
        try:
            self.collection.delete(where={"paper_id": paper_id})
            return True
        except Exception as e:
            print(f"Error deleting paper: {e}")
            return False
