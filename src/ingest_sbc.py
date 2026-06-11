from src.indexer import MultimodalIndexer
import sys

if __name__ == "__main__":
    force_reindex = "--reindex" in sys.argv or "-r" in sys.argv
    print(" Starting SBC Ingestion Pipeline")
    
    indexer = MultimodalIndexer(collection_name="sbc_collection", force_recreate=force_reindex)
    
    if force_reindex or indexer.is_collection_empty():
        indexer.index_folder("data/sbc", source_type="SBC")
        print(" SBC Indexing Completed!")
    else:
        print(" SBC collection already exists and is not empty.")
    
    indexer.close()