from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document

class BaseCSVLoader(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    @abstractmethod
    def load(self) -> List[Document]:
        """
        Load the CSV data and convert it to a list of LangChain Document objects.
        """
        pass
    