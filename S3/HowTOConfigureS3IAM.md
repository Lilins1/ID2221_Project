# AWS + RAG Python 开发环境搭建指南

本文档整理了从 AWS 账号准备到本地 Python 环境配置，以及 RAG（Retrieval-Augmented Generation）所需依赖安装的完整流程。

---

## 1. AWS 账号与 IAM 用户配置

### 1.1 创建 IAM 用户

1. 登录 [AWS 控制台](https://console.aws.amazon.com/iam/)。
2. 打开 **IAM → 用户 → 添加用户**。
3. 用户类型选择 **编程访问（Programmatic access / 本地代码）**。
4. 设置用户名，例如 `rag-dev-user`。
5. 权限选择：

   * **创建组**：例如 `S3AccessGroup`。
   * 附加策略：`AmazonS3FullAccess`（或 `AmazonS3ReadOnlyAccess`，只读）。
6. 用户创建完成后，记录 **Access Key ID** 和 **Secret Access Key**。

### 1.2 AWS CLI 配置

在本地机器或服务器上执行：

```bash
aws configure
```

依次输入：

* AWS Access Key ID
* AWS Secret Access Key
* 默认 region（如 `us-east-1`）
* 默认输出格式（json）

### 1.3 测试 S3 访问

```bash
# 列出所有 bucket
aws s3 ls

# 列出指定 bucket 中的文件
aws s3 ls s3://你的bucket名字

# 下载文件到本地
aws s3 cp s3://你的bucket名字/路径/文件.txt ./文件.txt
```

### 1.4 Python 访问 S3

```python
import boto3

s3 = boto3.client(
    's3',
    aws_access_key_id='你的AccessKey',
    aws_secret_access_key='你的SecretKey',
    region_name='us-east-1'
)

# 列出所有 bucket
for bucket in s3.list_buckets()['Buckets']:
    print(bucket['Name'])

# 下载文件
bucket_name = '你的bucket名字'
object_key = '路径/文件.txt'
s3.download_file(bucket_name, object_key, 'local_file.txt')
```

---

## 2. Conda 环境创建

### 2.1 创建新环境

```bash
conda create --name rag_env python=3.11
```

### 2.2 激活环境

```bash
conda activate rag_env
```

### 2.3 安装必需包

```bash
pip install boto3 awscli langchain openai sentence-transformers faiss-cpu chromadb pandas numpy tiktoken python-dotenv
```

> 注：如果有 GPU，可将 `faiss-cpu` 替换为 `faiss-gpu`。

### 2.4 验证安装

```python
import boto3
import langchain
import openai
import faiss
import chromadb
print('All packages installed successfully!')
```

---

## 3. 可选：挂载 S3 到本地目录

```bash
sudo apt install s3fs -y
echo ACCESS_KEY:SECRET_KEY > ~/.passwd-s3fs
chmod 600 ~/.passwd-s3fs
s3fs 你的bucket名字 /mnt/s3 -o passwd_file=~/.passwd-s3fs
```

现在 `/mnt/s3` 目录下可以直接访问 S3 文件。

---

## 4. JSON 配置示例

如果你希望在 Python 中用 JSON 配置访问 S3，可以创建 `config.json`：

```json
{
  "service": "s3",
  "aws_access_key_id": "你的AccessKey",
  "aws_secret_access_key": "你的SecretKey",
  "region_name": "us-east-1"
}
```

读取示例：

```python
import json
import boto3

with open('config.json') as f:
    cfg = json.load(f)

s3 = boto3.client(
    cfg['service'],
    aws_access_key_id=cfg['aws_access_key_id'],
    aws_secret_access_key=cfg['aws_secret_access_key'],
    region_name=cfg['region_name']
)
```

---

## 5. 总结

1. 创建 IAM 用户并分配 S3 访问权限。
2. 在本地配置 AWS CLI 或 Python 访问凭证。
3. 使用 Conda 创建隔离环境，并安装 AWS + RAG 相关依赖。
4. 可选挂载 S3 到本地方便操作。
5. JSON 配置可用于代码化管理凭证和服务配置。

这样就完成了 **AWS + RAG Python 开发环境** 的搭建。
