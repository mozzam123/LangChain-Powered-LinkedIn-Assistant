import pandas as pd
from typing import List
from langchain.docstore.document import Document
from .base_loader import BaseCSVLoader

class MessagesCSVLoader(BaseCSVLoader):
    def load(self) -> List[Document]:
        df = pd.read_csv(self.file_path)
        documents = []

        for _, row in df.iterrows():
            content = str(row.get("Content", "")).strip()
            metadata = {
                "from": row.get("From", ""),
                "to": row.get("To", ""),
                "date": row.get("Date", "")
            }

            if content:
                documents.append(Document(page_content=content, metadata=metadata))

        return documents
