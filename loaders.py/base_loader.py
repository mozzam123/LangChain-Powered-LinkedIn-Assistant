from abc import ABC, abstractmethod

class BaseCSVLoader(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    @abstractmethod
    def load(self):
        """
        Load the CSV data and convert it to a list of LangChain Document objects.
        """
        pass
    