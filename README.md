### Dataset curation
In order to utilise data they must be ingested

```bash
export PYTHONPATH="./src"
./bin/ingest_dataset -inputs ./data/nonpapal/* -output_dir ./data/ingested/nonpapal
./bin/ingest_dataset -inputs ./data/papal/* -output_dir ./data/ingested/papal
```


### TODO
* make archives (zip, rar, etc) ingestable for folder datasets
