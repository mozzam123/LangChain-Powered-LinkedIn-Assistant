import pandas as pd
from typing import List
from langchain.docstore.document import Document
from .base_loader import BaseCSVLoader

class ConnectionsCSVLoader(BaseCSVLoader):
    def load(self) -> List[Document]:
        df = pd.read_csv(self.file_path)
        documents = []

        for _, row in df.iterrows():
            full_name = f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip()
            company = row.get("Company", "")
            position = row.get("Position", "")

            content = f"{full_name} works at {company} as a {position}".strip()
            metadata = {
                "name": full_name,
                "company": company,
                "position": position
            }

            documents.append(Document(page_content=content, metadata=metadata))

        return documents
