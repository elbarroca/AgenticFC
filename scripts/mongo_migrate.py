# %% [markdown]
# # MongoDB to Couchbase Migration Script

# %%
import certifi
from pymongo import MongoClient
from couchbase.cluster import Cluster, ClusterOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.collection import UpsertOptions
from couchbase.exceptions import DocumentExistsException
import uuid
import datetime
from bson import ObjectId

# --- CONFIGURATION ---
# MongoDB Source
SRC_URI = "mongodb+srv://test:test@cluster0.eqpdcza.mongodb.net/"
SRC_DB = "agenticfc"

# Couchbase Destination
CB_CONN_STR = "couchbases://cb.9ns2cxt3drqtl0v.cloud.couchbase.com"
CB_USERNAME = "Admin888"
CB_PASSWORD = "Admin888?"
CB_BUCKET = "agenticfc"  # e.g., "default"

# %% [markdown]
# ## Connect to MongoDB

# %%
src_client = MongoClient(
    SRC_URI,
    tls=True,
    tlsCAFile=certifi.where()
)
src_db = src_client[SRC_DB]

# %% [markdown]
# ## Connect to Couchbase

# %%
cluster = Cluster(
    CB_CONN_STR,
    ClusterOptions(PasswordAuthenticator(CB_USERNAME, CB_PASSWORD))
)
bucket = cluster.bucket(CB_BUCKET)
# bucket.on_connect()  # Ensures connection is ready

# Optionally, use a named collection (Couchbase 7+)
# collection = bucket.scope("myscope").collection("mycollection")
collection = bucket.default_collection()

# %% [markdown]
# ## Migrate Collections

# %%
collections = src_db.list_collection_names()
print(f"Collections to migrate: {collections}")

def convert_bson(obj):
    if isinstance(obj, dict):
        return {k: convert_bson(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_bson(i) for i in obj]
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, ObjectId):
        return str(obj)
    else:
        return obj

for coll_name in collections:
    print(f"Migrating collection: {coll_name}")
    src_coll = src_db[coll_name]
    docs = list(src_coll.find({}))
    migrated = 0

    # Use the collection with the same name as in MongoDB, under the default scope
    cb_collection = bucket.scope("_default").collection(coll_name)

    for doc in docs:
        doc_id = str(doc.get("_id", uuid.uuid4()))
        doc.pop("_id", None)
        doc = convert_bson(doc)
        try:
            cb_collection.upsert(doc_id, doc)
            migrated += 1
        except DocumentExistsException:
            print(f"  Document {doc_id} already exists, skipping.")
    print(f"  Migrated {migrated} documents.")

print("Migration complete.")

# %% [markdown]
# ## Close connections

# %%
src_client.close()
cluster.close()