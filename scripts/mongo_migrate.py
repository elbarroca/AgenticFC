
import certifi
from pymongo import MongoClient
import uuid
import datetime
from bson import ObjectId

# --- CONFIGURATION ---
# MongoDB Source
SRC_URI = "mongodb://admin888:admin888@127.0.0.1:27017/?authSource=admin"
SRC_DB_NAME = "agenticfc"  # Explicitly naming for clarity

# MongoDB Destination (New)
DEST_URI = "mongodb://root:RicardoMongoDB@74.50.127.165:27017/admin" # /admin for authSource
DEST_DB_NAME = "agenticfc" # Target database name on the new server

# ## Connect to MongoDB
src_client = MongoClient(
    SRC_URI
)
src_db = src_client[SRC_DB_NAME]

# Connect to MongoDB Destination
dest_client = MongoClient(DEST_URI)
dest_db = dest_client[DEST_DB_NAME]

# ## Migrate Collections
collections = src_db.list_collection_names()
print(f"Collections to migrate: {collections}")

for coll_name in collections:
    print(f"Migrating collection: {coll_name}")
    src_coll = src_db[coll_name]
    dest_coll = dest_db[coll_name] # Get collection in destination
    
    migrated_count = 0
    
    # Iterate over the cursor to avoid loading all documents into memory
    for doc_to_migrate in src_coll.find({}):
        try:
            # Preserve the original document, including its _id and BSON types.
            dest_coll.replace_one({'_id': doc_to_migrate['_id']}, doc_to_migrate, upsert=True)
            migrated_count += 1
        except Exception as e: # Consider more specific error handling if needed
            print(f"  Error migrating document {doc_to_migrate.get('_id', 'UNKNOWN_ID')} in {coll_name}: {e}")
            
    print(f"  Migrated {migrated_count} documents to collection {coll_name} in destination MongoDB.")

print("Migration complete.")

# ## Close connections
src_client.close()
dest_client.close() # Close destination client