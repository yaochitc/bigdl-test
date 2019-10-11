package io.yaochi.model

import com.intel.analytics.bigdl.nn.{LookupTable, Sequential, Sigmoid}
import com.intel.analytics.bigdl.nn.keras.{Embedding, Input, Model}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers.KerasLayerWrapper
import io.yaochi.nn.{WeightedMerge, WeightedMergerLayer}

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

  def buildKeras(): Model[T] = {
    val input = Input(Shape(featureNum))
    val embedding = Embedding[T](vocabSize, featureDim).inputs(input)
    val merged = new WeightedMergerLayer[T]().inputs(embedding)
    val ouput = new KerasLayerWrapper[T](Sigmoid[T]()).inputs(merged)
    Model(input, ouput)
  }
}
