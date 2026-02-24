from doc_extracter import DocumentIngestionPipeline

file_path = "data/ChatGPT Image Dec 12, 2025, 08_42_59 AM.png"

pipeline = DocumentIngestionPipeline()
result = pipeline.ingest(file_path)

print(result.summary())

print(result.plain_text)

