# RUN: %python %s nanobind | FileCheck %s

from mlir_spark.ir import *
from mlir_spark.dialects import spark_nanobind as spark_d

with Context():
    spark_d.register_dialects()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = spark.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: spark.foo %[[C]] : i32
    print(str(module))
