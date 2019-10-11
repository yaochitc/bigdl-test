package io.yaochi.nn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Shape, SingleShape, T, Table}

import scala.reflect.ClassTag

class WeightedMergerLayer[T: ClassTag]
(val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](WeightedMerge.addBatch(inputShape)) {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    new WeightedMerge[T](input(1))
  }

  override def clearState(): this.type = {
    if (output.isInstanceOf[Tensor[_]]) {
      output = Tensor[T]()
    }

    if (gradInput.isInstanceOf[Tensor[_]]) {
      gradInput = Tensor[T]()
    }

    this
  }

  override def computeOutputShape(calcInputShape: Shape): Shape = {
    val input = calcInputShape.toSingle().toArray
    require(input.length == 3,
      s"Embedding requires 3D input, but got input dim ${input.length}")
    Shape(input(0), input(2))
  }

}

class WeightedMerge[T: ClassTag](size: Int)
                                (implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  val weight: Tensor[T] = Tensor[T](size)

  val gradWeight: Tensor[T] = Tensor[T](size)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 3,
      "3D tensor expected" +
        s"input dimension ${input.nDimension()}")

    output.resize(input.size(1), input.size(3)).zero()
    val (nFrame, dim, stride) = (input.size(1), input.size(2), input.size(3))

    val outputArray = output.storage().array()
    val inputArray = if (input.isContiguous()) {
      input.storage().array()
    } else {
      input.contiguous().storage().array()
    }
    val storageOffset = input.storageOffset() - 1

    val scales = (1 to size).map(i => ev.exp(weight.valueAt(i)))
    val scaleSum = scales.reduceLeft((a, b) => ev.plus(a, b))
    val normScales = scales.map(scale => ev.divide(scale, scaleSum))

    var i = 0
    while (i < nFrame) {
      var d = 0
      while (d < dim) {
        var j = 0
        while (j < stride) {
          val inputOffset = i * dim * stride + d * stride + j + storageOffset
          val outputOffset = i * stride + j

          val z = ev.times(inputArray(inputOffset), normScales(d))
          outputArray(outputOffset) = ev.plus(outputArray(outputOffset), z)
          j += 1
        }
        d += 1
      }
      i += 1
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (!gradInput.isSameSizeAs(input)) {
      gradInput.resizeAs(input).zero()
    }
    val (nFrame, dim, stride) = (input.size(1), input.size(2), input.size(3))

    val gradInputArray = gradInput.storage().array()
    val gradOutputArray = if (gradOutput.isContiguous()) {
      gradOutput.storage().array()
    } else {
      gradOutput.contiguous().storage().array()
    }

    val scales = (1 to size).map(i => ev.exp(weight.valueAt(i)))
    val scaleSum = scales.reduceLeft((a, b) => ev.plus(a, b))
    val normScales = scales.map(scale => ev.divide(scale, scaleSum))
    val squareNormScales = normScales.map(scale => ev.times(scale, scale))

    var i = 0
    while (i < nFrame) {
      var d = 0
      while (d < dim) {
        var j = 0
        while (j < stride) {
          val gradInputOffset = i * dim * stride + d * stride + j
          val gradOutputOffset = i * stride + j

          gradInputArray(gradInputOffset) = ev.times(gradOutputArray(gradOutputOffset), squareNormScales(d))
          j += 1
        }
        d += 1
      }
      i += 1
    }

    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    gradWeight.resize(size)
    val (nFrame, dim, stride) = (input.size(1), input.size(2), input.size(3))

    val inputArray = if (input.isContiguous()) {
      input.storage().array()
    } else {
      input.contiguous().storage().array()
    }
    val gradWeightArray = gradWeight.storage().array()
    val gradWeightOffset = gradWeight.storageOffset() - 1
    val gradOutputArray = if (gradOutput.isContiguous()) {
      gradOutput.storage().array()
    } else {
      gradOutput.contiguous().storage().array()
    }
    val storageOffset = input.storageOffset() - 1

    val scales = (1 to size).map(i => ev.exp(weight.valueAt(i)))
    val scaleSum = scales.reduceLeft((a, b) => ev.plus(a, b))
    val normScales = scales.map(scale => ev.divide(scale, scaleSum))
    val squareNormScales = normScales.map(scale => ev.times(scale, scale))

    var i = 0
    while (i < nFrame) {
      var d = 0
      while (d < dim) {
        var j = 0
        while (j < stride) {
          val inputOffset = i * dim * stride + d * stride + j + storageOffset
          val gradOutputOffset = i * stride + j

          val z = ev.times(inputArray(inputOffset), ev.times(gradOutputArray(gradOutputOffset), squareNormScales(d)))
          gradWeightArray(d + gradWeightOffset) = ev.plus(gradWeightArray(d + gradWeightOffset), z)
          j += 1
        }
        d += 1
      }
      i += 1
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight), Array(this.gradWeight))
  }

  override def getParametersTable(): Table = {
    T(getName() -> T("weight" -> weight, "gradWeight" -> gradWeight))
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[WeightedMerge[T]]) {
      return false
    }
    val other = obj.asInstanceOf[WeightedMerge[T]]
    if (this.eq(other)) {
      return true
    }

    gradWeight == other.gradWeight &&
      weight == other.weight
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + gradWeight.hashCode()
    hash = hash * seed + weight.hashCode()

    hash
  }
}

object WeightedMerge {
  def addBatch(shape: Shape): Shape = {
    // simply return null here as null is the default value
    if (shape == null) {
      return null
    }
    if (shape.isInstanceOf[SingleShape]) {
      Shape((List(-1) ++ shape.toSingle()).toArray)
    } else {
      Shape(shape.toMulti().map {
        addBatch(_)
      })
    }
  }
}
