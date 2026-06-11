from src.indexer import MultimodalIndexer
import sys

if __name__ == "__main__":
    force_reindex = "--reindex" in sys.argv or "-r" in sys.argv
    print(" Starting SPD Ingestion Pipeline")
    
    indexer = MultimodalIndexer(collection_name="spd_collection", force_recreate=force_reindex)
    
    if force_reindex or indexer.is_collection_empty():
        indexer.index_folder("data/spd", source_type="SPD")
        print(" SPD Indexing Completed!")
    else:
        print(" SPD collection already exists and is not empty.")
    
    indexer.close()