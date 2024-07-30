
#  ДАТАСЕТ В ФОРМАТ ДОКУМЕНТА
class DataFrameLoader:
    def __init__(self, df, page_content_column):
        self.df = df
        self.page_content_column = page_content_column

    def load(self):
        return [Document(text) for text in self.df[self.page_content_column].tolist()]


class Document:
    def __init__(self, page_content):
        self.page_content = page_content


# Разделяем документы на части
class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents):
        texts = []
        for document in documents:
            text = document.page_content
            for i in range(0, len(text), self.chunk_size):
                end = i + self.chunk_size
                if end >= len(text):
                    texts.append(text[i:])
                else:
                    texts.append(text[i:end + self.chunk_overlap])
        return texts
