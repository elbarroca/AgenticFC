from pymongo import MongoClient, errors
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

# ## Connect to MongoDB Source
try:
    src_client = MongoClient(
        SRC_URI,
        serverSelectionTimeoutMS=5000 # Timeout for server selection
    )
    # Verify connection
    src_client.admin.command('ping')
    print("Successfully connected to Source MongoDB.")
    src_db = src_client[SRC_DB_NAME]
except errors.ConnectionFailure as e:
    print(f"Could not connect to Source MongoDB: {e}")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred with Source MongoDB connection: {e}")
    exit(1)


# ## Connect to MongoDB Destination
try:
    dest_client = MongoClient(
        DEST_URI,
        serverSelectionTimeoutMS=5000
    )
    # Verify connection
    dest_client.admin.command('ping')
    print("Successfully connected to Destination MongoDB.")
    dest_db = dest_client[DEST_DB_NAME]
except errors.ConnectionFailure as e:
    print(f"Could not connect to Destination MongoDB: {e}")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred with Destination MongoDB connection: {e}")
    # Clean up source client if destination fails
    if 'src_client' in locals() and src_client:
        src_client.close()
    exit(1)

# ## Migrate Collections
try:
    collections = src_db.list_collection_names()
    print(f"Collections found in source: {collections}")

    for coll_name in collections:
        print(f"\nProcessing collection: {coll_name}")
        src_coll = src_db[coll_name]
        dest_coll = dest_db[coll_name] # Get collection in destination

        # Optional: Create indexes on destination if they don't exist
        # This is good practice but can slow down initial bulk inserts.
        # You might prefer to create indexes after the migration.
        # try:
        #     src_indexes = src_coll.index_information()
        #     for index_name, index_info in src_indexes.items():
        #         if index_name == "_id_": # Skip default _id index
        #             continue
        #         keys = index_info['key']
        #         # Remove 'v' if present, as create_index doesn't use it directly
        #         index_options = {k: v for k, v in index_info.items() if k not in ['key', 'v', 'ns']}
        #         print(f"  Ensuring index '{index_name}' on {keys} with options {index_options} in destination.")
        #         dest_coll.create_index(keys, name=index_name, **index_options)
        # except Exception as e:
        #     print(f"  Warning: Could not replicate indexes for {coll_name}: {e}")


        inserted_count = 0
        skipped_count = 0
        error_count = 0

        # Iterate over the cursor to avoid loading all documents into memory
        # Consider adding a query filter to src_coll.find() if you only want to
        # process documents newer than a certain date/ObjectId, for example.
        # e.g., src_coll.find({'_id': {'$gt': last_migrated_object_id}})
        # For a full "only new" based on destination, the check below is needed.

        for doc_to_migrate in src_coll.find({}):
            doc_id = doc_to_migrate.get('_id')
            if doc_id is None:
                print(f"  Skipping document without _id in {coll_name}: {doc_to_migrate}")
                error_count +=1
                continue

            try:
                # Check if document with this _id already exists in destination
                existing_doc = dest_coll.find_one({'_id': doc_id})

                if existing_doc is None:
                    # Document does not exist in destination, so insert it
                    dest_coll.insert_one(doc_to_migrate)
                    inserted_count += 1
                else:
                    # Document already exists in destination, skip it
                    skipped_count += 1

            except errors.DuplicateKeyError:
                # This should ideally not happen if find_one check is robust,
                # but can occur in concurrent scenarios or if _id was somehow missed by find_one.
                print(f"  Skipped document {doc_id} in {coll_name} due to DuplicateKeyError (already exists).")
                skipped_count += 1
            except errors.WriteConcernError as wce:
                print(f"  Write concern error migrating document {doc_id} in {coll_name}: {wce}")
                error_count +=1
            except errors.PyMongoError as pme: # Catch other PyMongo specific errors
                print(f"  PyMongo error migrating document {doc_id} in {coll_name}: {pme}")
                error_count +=1
            except Exception as e: # Catch any other unexpected errors
                print(f"  Unexpected error migrating document {doc_id} in {coll_name}: {e}")
                error_count +=1

        print(f"  Collection {coll_name}:")
        print(f"    Inserted {inserted_count} new documents.")
        print(f"    Skipped {skipped_count} existing documents.")
        if error_count > 0:
            print(f"    Encountered {error_count} errors.")

    print("\nMigration process complete.")

except errors.OperationFailure as op_e:
    print(f"A MongoDB operation failed during migration: {op_e}")
except Exception as e:
    print(f"An unexpected error occurred during migration: {e}")
finally:
    # ## Close connections
    if 'src_client' in locals() and src_client:
        src_client.close()
        print("Source MongoDB connection closed.")
    if 'dest_client' in locals() and dest_client:
        dest_client.close()
        print("Destination MongoDB connection closed.")