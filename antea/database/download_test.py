import os

import sqlite3
import pymysql
pymysql.install_as_MySQLdb()

from . import download as db


def test_create_table_sqlite(output_tmpdir):
    dbname = 'PETALODB'
    dbfile = os.path.join(output_tmpdir, 'db.sqlite3')

    if os.path.isfile(dbfile):
        os.remove(dbfile)

    connSqlite = sqlite3.connect(dbfile)
    connMySql  = pymysql.connect(host="next.ific.uv.es",
                                user='petaloreader',passwd='petaloreader', db=dbname)

    cursorMySql  = connMySql .cursor()
    cursorSqlite = connSqlite.cursor()

    for table in db.tables:
        db.create_table_sqlite(cursorSqlite, cursorMySql, table)
