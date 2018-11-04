# Feature

# Run as Docker

```bash
docker run --name tf-recog-number -itd -p 5000:5000 tf-recog-number:0.1
```

# Request Sample

```bash
curl -X POST -H "Content-type: application/json" -d@sample_reqs/req7.json localhost:5000/recognize
```

