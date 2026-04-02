from pipeline.rag_pipeline import AdvancedRAGPipeline
from config import Config

if __name__ == "__main__":
    config = Config()
    pipeline = AdvancedRAGPipeline(config=config)  
    pipeline.ingest("data/")
    result = pipeline.run("Your question here?")
    print(result["answer"])