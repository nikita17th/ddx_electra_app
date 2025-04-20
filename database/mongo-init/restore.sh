#!/bin/bash

until mongosh --eval "print(\"MongoDB is ready\")" 2>/dev/null; do
  sleep 2
done

mongorestore --uri="mongodb://root:example@localhost:27017/survey_db?authSource=admin" --dir=/docker-entrypoint-initdb.d/dump/survey_db
