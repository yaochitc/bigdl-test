package io.yaochi.model

import com.intel.analytics.bigdl.nn.{LookupTable, Sequential, Sigmoid}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import io.yaochi.nn.WeightedMerge

import scala.reflect.ClassTag

class ExampleModel[T: ClassTag](featureNum: Int,
                                vocabSize: Int,
                                featureDim: Int)
                               (implicit ev: TensorNumeric[T]) {
  def build(): Sequential[T] = {
    val model = Sequential[T]()

    model.add(LookupTable[T](vocabSize, featureDim))
      .add(new WeightedMerge[T](featureNum))
      .add(Sigmoid[T]())
    model
  }
}
