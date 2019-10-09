package io.yaochi.model

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import io.yaochi.nn.WeightedMerge

import scala.reflect.ClassTag

class ExampleModel[T: ClassTag](featureNum: Int,
                                vocabSize: Int,
                                featureDim: Int)
                               (implicit ev: TensorNumeric[T]) {
  def build(): Model[T] = {
    val input = Input[T](inputShape = Shape(featureNum))
    val wordEmbedding = Embedding[T](vocabSize, featureDim).inputs(input)

    val merged = new KerasLayerWrapper[T](new WeightedMerge[T](featureNum)).inputs(wordEmbedding)
    val logits = Dense[T](1, activation = "sigmoid").inputs(merged)
    Model[T](input, logits)
  }
}
