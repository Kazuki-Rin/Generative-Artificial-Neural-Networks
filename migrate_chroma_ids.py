import os
import hashlib
import chromadb
import time

def migrate():
    print("LEGACY UUID PRESERVATION: NOT POSSIBLE FROM AVAILABLE DATA")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_id = "M8_Memory"
    db_path = os.path.join(base_dir, "models", model_id, "chroma_db")
    if not os.path.exists(db_path):
        print("No chroma db found")
        return
        
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="episodic_memory")
    
    all_data = collection.get(include=["embeddings", "metadatas", "documents"])
    ids = all_data.get("ids", [])
    if not ids:
        print("ChromaDB is empty.")
        return

    embeddings = all_data.get("embeddings", [])
    metadatas = all_data.get("metadatas", [])
    documents = all_data.get("documents", [])

    new_ids = []
    new_embeddings = []
    new_metadatas = []
    new_documents = []
    ids_to_delete = []

    mapping_log = []
    
    for i in range(len(ids)):
        old_id = ids[i]
        emb = embeddings[i]
        meta = metadatas[i] if metadatas and metadatas[i] else {}
        doc = documents[i] if documents and documents[i] else ""
        
        if not old_id.startswith("migrated_") and not meta.get("migrated_from"):
            migration_id = f"migrated_{old_id}"
            
            emb_str = ",".join(f"{float(x):.5f}" for x in emb)
            fingerprint = hashlib.md5(emb_str.encode('utf-8')).hexdigest()
            
            if migration_id not in ids and migration_id not in new_ids:
                new_meta = meta.copy()
                new_meta["legacy_id"] = old_id
                new_meta["migration_version"] = "1.0"
                new_meta["migration_timestamp"] = str(time.time())
                new_meta["source_record_fingerprint"] = fingerprint
                new_meta["migrated_from"] = True
                
                if "art_node" not in new_meta or not str(new_meta["art_node"]).startswith("legacy_"):
                    new_meta["art_node"] = f"legacy_{fingerprint[:16]}"
                
                new_ids.append(migration_id)
                new_embeddings.append(emb)
                new_metadatas.append(new_meta)
                new_documents.append(doc)
                ids_to_delete.append(old_id)
                
                mapping_log.append(f"{old_id} -> {migration_id} (ART: {new_meta['art_node']})")

    if new_ids:
        collection.add(
            embeddings=new_embeddings,
            metadatas=new_metadatas,
            documents=new_documents,
            ids=new_ids
        )
        # [BẢN VÁ P0]: Xóa bản ghi cũ để tránh duplicate dữ liệu
        collection.delete(ids=ids_to_delete)
        print(f"Added {len(new_ids)} migrated legacy records. Old records were DELETED to prevent duplication.")
        
        with open("chroma_migration_mapping.log", "a") as f:
            f.write("\n".join(mapping_log) + "\n")
        print("Migration mapping saved to chroma_migration_mapping.log")
    else:
        print("No new heuristic records needed.")

if __name__ == "__main__":
    migrate()
