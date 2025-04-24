# %% [markdown]
# # MongoDB Migration Script
# This script copies all collections and documents from the source MongoDB to the destination MongoDB.

# %%
import certifi
from pymongo import MongoClient

# --- CONFIGURATION ---
# Source (current Atlas cluster)
SRC_URI = "mongodb+srv://test:test@cluster0.eqpdcza.mongodb.net/"
SRC_DB = "agenticfc"

# Destination (new server)
DEST_URI = "mongodb://admin888:admin888@127.0.0.1:27017/?authSource=admin"
DEST_DB = "agenticfc"  # Use the same DB name unless you want to change

# %% [markdown]
# ## Connect to both databases

# %%
src_client = MongoClient(
    SRC_URI,
    tls=True,
    tlsCAFile=certifi.where()
)
src_db = src_client[SRC_DB]

dest_client = MongoClient(DEST_URI)
dest_db = dest_client[DEST_DB]

# %% [markdown]
# ## List all collections and migrate

# %%
collections = src_db.list_collection_names()
print(f"Collections to migrate: {collections}")

for coll_name in collections:
    print(f"Migrating collection: {coll_name}")
    src_coll = src_db[coll_name]
    dest_coll = dest_db[coll_name]
    
    # Remove all documents in the destination collection (optional, for full replacement)
    dest_coll.delete_many({})
    
    # Fetch all documents from source
    docs = list(src_coll.find({}))
    if docs:
        # Insert into destination
        dest_coll.insert_many(docs)
        print(f"  Migrated {len(docs)} documents.")
    else:
        print("  No documents to migrate.")

print("Migration complete.")

# %% [markdown]
# ## Close connections

# %%
src_client.close()
dest_client.close()