import json
import boto3

# 假设 json 配置保存在 config.json
with open("config.json") as f:
    cfg = json.load(f)

s3 = boto3.client(
    cfg["service"],
    aws_access_key_id=cfg["aws_access_key_id"],
    aws_secret_access_key=cfg["aws_secret_access_key"],
    region_name=cfg["region_name"]
)

# 测试
for bucket in s3.list_buckets()["Buckets"]:
    print(bucket["Name"])
    response = s3.list_objects_v2(Bucket=bucket["Name"])

    if 'Contents' in response:
        for obj in response['Contents']:
            print(obj['Key'])
    else:
        print("Bucket is empty")