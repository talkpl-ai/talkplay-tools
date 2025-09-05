"""SQLite-based boolean retrieval over a tracks table for filtering/rules queries."""
import os
import json
import sqlite3
import sqlparse
import datetime

class SQLTool:
    """Query a SQLite database of tracks with schema-aware helpers.

    Args:
        cache_dir (str): Directory containing SQL DB and metadata files.
    """
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = cache_dir
        self.retriever = self.load_model()
        if self.retriever is None:
            self.build_index()
        self._init_track_pool()

    def _init_track_pool(self):
        """Reset the internal track pool restriction."""
        self._track_pool = None

    def _update_track_pool(self, track_pool: list[str]):
        """Restrict retrieval to the provided candidate track ids."""
        self._track_pool = track_pool

    def load_model(self):
        """Load or return the SQLite connection if present."""
        db_path = os.path.join(self.cache_dir, "sql", "tracks.db")
        if not os.path.exists(db_path):
            return None
        return sqlite3.connect(db_path)

    def sql_db_schema(self):
        """Return a description string of the DB schema with simple stats/examples."""
        base_description = "track_id TEXT PRIMARY KEY, title TEXT, artist TEXT, album TEXT, popularity INTEGER, release_date DATETIME (YYYY-MM-DD), tempo REAL, key TEXT"
        cursor = self.retriever.cursor()
        db_information = {}
        for col in ['title', 'artist', 'album']:
            cursor.execute(f"SELECT DISTINCT {col} FROM tracks WHERE {col} IS NOT NULL ORDER BY {col}")
            examples = cursor.fetchmany(2)
            examples = ", ".join([i[0] for i in examples])
            db_information[col] = {
                "description": f"{col} column is string data type, the examples are {examples}",
                "type": "string"
            }
        for col in ['popularity', 'release_date', 'tempo']:
            cursor.execute(f"SELECT MIN({col}), MAX({col}), AVG({col}) FROM tracks")
            stats = cursor.fetchone()
            if col == 'release_date':
                desc = f"{col} column is datetime data type, the range is from 1900-01-01 to {stats[1]}, the average is {stats[2]:.2f}"
            else:
                desc = f"{col} column is integer data type, the range is from {stats[0]} to {stats[1]}, the average is {stats[2]:.2f}"
            db_information[col] = {
                "description": desc,
                "type": "integer"
            }
        cursor.execute("SELECT DISTINCT key FROM tracks WHERE key IS NOT NULL ORDER BY key")
        unique_keys = cursor.fetchall()
        unique_keys = [i[0] for i in unique_keys]
        db_information['key'] = {
            "description": f"key column is string data type, the unique values are {unique_keys}",
            # "examples": key_examples,
            "type": "string"
        }
        return f"{base_description}\n{db_information}"

    def apply_datetime(self, date_str: str):
        """Convert YYYY-MM-DD to a Unix timestamp (seconds)."""
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").timestamp()

    def build_index(self):
        """Create and populate the SQLite DB from cached metadata JSON."""
        test_metadata = json.load(open(os.path.join(self.cache_dir, "metadata", "test_metadata.json"), "r"))
        # Create SQLite database
        db_path = os.path.join(self.cache_dir, "sql", "tracks.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Create tracks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracks (
                track_id TEXT PRIMARY KEY,
                title TEXT,
                artist TEXT,
                album TEXT,
                popularity INTEGER,
                release_date DATETIME,
                tempo REAL,
                key TEXT
            )
        ''')
        # Insert track data
        for k,v in test_metadata.items():
            cursor.execute('''
                INSERT OR REPLACE INTO tracks (track_id, title, artist, album, popularity, release_date, tempo, key)
                VALUES (?,?,?,?,?,?,?,?)
            ''', (
                v['track_id'],
                ",".join(v['track_name']),
                ",".join(v['artist_name']),
                ",".join(v['album_name']),
                v['popularity'],
                v['track_release_date_spotify'],
                float(v['tempo'][0]) if len(v['tempo']) > 0 else None,
                v['key'][0] if len(v['key']) > 0 else None
            ))
        conn.commit()
        conn.close()
        self.retriever = sqlite3.connect(db_path)

    def sql_retrieval(self, query: str, topk: int):
        """Execute an SQL query and filter by the current track pool.

        The input `query` must return a `track_id` column.

        Args:
            query (str): SQL statement (SELECT ... FROM tracks ...).
            topk (int): Maximum number of results.

        Returns:
            list[str]: Matching track ids.
        """
        cursor = self.retriever.cursor()
        try:
            base_q = str(sqlparse.parse(query)[0]).strip().rstrip(";")
            placeholders = ",".join(["?"] * len(self._track_pool))
            final_q = f"SELECT track_id FROM ({base_q}) AS t WHERE track_id IN ({placeholders})"
            params = list(self._track_pool)
            cursor.execute(final_q, params)
            result = cursor.fetchall()
            track_ids = [i[0] for i in result]
            return track_ids[:topk]
        except Exception as e:
            raise ValueError(f"Error in sql_retrieval: {e}")
