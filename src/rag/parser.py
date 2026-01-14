from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader

loader = OpenDataLoaderPDFLoader(
    file_path=[r"E:\LLMOps\agentic-rag\data\raw\Employees' Provident Funds Scheme.1952.pdf"],
    format="json", 
    quiet=True
)
documents = loader.load()
print(len(documents))
print(documents[43].metadata)
print((documents[43].page_content))

# for doc in documents:
#     print(doc.metadata, doc.page_content[:80])