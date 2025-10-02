

第 2 步：准备最终的 Spark 脚本
我们已经将所有修正和增强功能整合到了 Canvas 中的 process_xml_to_json.py 脚本中。

脚本位置: 确保该脚本位于您本地的 ./app-code 文件夹中。

关键功能:

健壮的字段提取: 能正确处理包含特殊字符（如 -）的字段名。

智能的正文处理: 能够处理复杂的段落标签，避免因数据类型不一致导致的错误。

优化的分类逻辑: 能处理类别名称中可能出现的 [] 字符，并将其正确转换为目录名。

第 3 步：执行完整、正确的 spark-submit 命令
有了正确的环境和脚本，现在我们可以使用最终的命令来提交任务。

在您的本地电脑终端（PowerShell 或 Bash），运行以下命令：

docker exec spark-cluster spark-submit \
  --packages com.databricks:spark-xml_2.12:0.17.0 \
  --conf "spark.ivy.home=/tmp" \
  /opt/spark/app/process_xml_to_json.py

命令解释:

--packages ...: 加载 Spark 解析 XML 所需的外部依赖包。

--conf "spark.ivy.home=/tmp": 这是一个重要的安全措施。它强制 Spark 将下载的依赖包缓存到容器内的 /tmp 目录，彻底杜绝了任何可能因主目录不明确而引发的路径错误。

/opt/spark/app/process_xml_to_json.py: 指定要执行的、功能完备的脚本。

第 4 步：验证结果
任务成功完成后，您可以按照以下步骤验证输出的 JSON 文件。

进入 Spark 容器：

docker exec -it spark-cluster bash

查看 HDFS 上按类别生成的目录：

hdfs dfs -ls /id2221/processed_json

查看其中一个 JSON 文件的内容来验证其结构和数据：

# 注意：路径和文件名需要替换成您实际看到的
hdfs dfs -cat /id2221/processed_json/category_partition=editorial/part-....json
