from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, lower, regexp_replace, transform, array_join, make_date

def main():
    """
    Main function to run the Spark job for converting XML to structured, 
    categorized JSON, with error handling for missing fields.
    """
    # =========================================================================
    #  用户配置区 (User Configuration)
    # =========================================================================
    
    # 1. XML 记录的根标签
    xml_row_tag = 'article' 

    # 2. 各个字段在 XML 中的路径 (使用点.分隔嵌套标签)
    pmc_id_path = "front.article-meta.article-id"
    category_path = "front.article-meta.article-categories.subj-group"
    title_path = "front.article-meta.title-group.article-title"
    author_path = "front.article-meta.contrib-group.contrib"
    pub_date_path = "front.article-meta.pub-date"
    body_path = "body.p"

    # HDFS 输入和输出路径
    input_path = "hdfs://namenode:9000/id2221/raw_xml_8M"
    output_path = "hdfs://namenode:9000/id2221/processed_json"

    # 初始化 SparkSession
    spark = SparkSession.builder \
        .appName("XML to Structured JSON for RAG") \
        .getOrCreate()

    print("SparkSession created successfully.")
    print(f"Reading XML files from: {input_path}")

    try:
        # 1. 读取 XML 数据
        # (最佳实践优化: 使用 'xml' 作为 format 的别名)
        df = spark.read \
            .format("com.databricks.spark.xml") \
            .option("rowTag", xml_row_tag) \
            .load(input_path)

        print("XML data loaded. Initial Schema:")
        df.printSchema(level=3)

        # 2. 提取和转换字段
        processed_df = df.select(
            col(pmc_id_path).getItem(1).getField("_VALUE").alias("pmc_id"),
            col(category_path).getItem(0).getField("subject").alias("category"),
            col(title_path).alias("title"),
            
            transform(
                col(author_path),
                lambda x: concat_ws(
                    ", ",
                    x.getField("name").getField("surname"),  # 去掉下划线
                    x.getField("name").getField("given-names")  # 去掉下划线
                )
            ).alias("authors"),
            
            make_date(
                col(pub_date_path).getItem(0).getField("year"),
                col(pub_date_path).getItem(0).getField("month"),
                col(pub_date_path).getItem(0).getField("day")
            ).alias("publication_date"),
            
            # 正文字段（修复类型不匹配问题）
            array_join(
                transform(
                    col(body_path),
                    lambda x: concat_ws(" ", x.getField("_VALUE"))  # 提取_VALUE并转为字符串
                ), 
                "\n\n"
            ).alias("body_text")
        )
        
        # 3. 创建用于分区的类别列
        processed_df = processed_df.withColumn(
            "category_partition",
            # 移除方括号[]，并将空格/斜杠替换为下划线
            lower(
                regexp_replace(
                    regexp_replace(col("category").cast("string"), "[\\[\\]]", ""),  # 先移除[]
                    "[\\s/]+", "_"  # 再替换空格/斜杠为下划线
                )
            )
        ).na.fill({"category_partition": "unknown"})


        print("DataFrame sample after processing:")
        processed_df.show(5, truncate=True)
        print("Final Schema before saving:")
        processed_df.printSchema()

        # 4. 按类别分区并保存为 JSON 文件
        print(f"Extracted and structured data. Saving to: {output_path}")
        processed_df.write \
            .partitionBy("category_partition") \
            .format("json") \
            .mode("overwrite") \
            .save(output_path)

        print("Job completed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        spark.stop()
        print("SparkSession stopped.")


if __name__ == "__main__":
    main()

