package io.yaochi.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class WeightedMerge[T: ClassTag](size: Int)
                                (implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  val weight: Tensor[T] = Tensor[T](size)

  val gradWeight: Tensor[T] = Tensor[T](size)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 3,
      "3D tensor expected" +
        s"input dimension ${input.nDimension()}")

    output.resize(input.size(1), input.size(3))
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

    var t = 0
    while (t < stride * nFrame) {
      val inputOffset = (t / stride) * dim * stride + t % stride + storageOffset
      val outputOffset = (t / stride) * dim * stride + t % stride

      var d = 0
      while (d < dim) {
        val z = ev.times(inputArray(d * stride + inputOffset), normScales(d))
        outputArray(outputOffset) = ev.plus(outputArray(outputOffset), z)
        d += 1
      }
      t += 1
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val nElement = gradInput.nElement()
    gradInput.resizeAs(input)
    if (nElement != gradInput.nElement()) {
      gradInput.zero()
    }
    val (nFrame, dim, stride) = (input.size(1), input.size(2), input.size(3))

    val gradInputArray = gradInput.storage().array()
    val outputArray = if (output.isContiguous()) {
      output.storage().array()
    } else {
      output.contiguous().storage().array()
    }
    val gradOutputArray = if (gradOutput.isContiguous()) {
      gradOutput.storage().array()
    } else {
      gradOutput.contiguous().storage().array()
    }

    var t = 0
    while (t < stride * nFrame) {
      var d = 0
      while (d < dim) {
        d += 1
      }
      t += 1
    }

    gradInput
  }


  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    val (nFrame, dim, stride) = (input.size(1), input.size(2), input.size(3))

    val outputArray = if (output.isContiguous()) {
      output.storage().array()
    } else {
      output.contiguous().storage().array()
    }
    val gradOutputArray = if (gradOutput.isContiguous()) {
      gradOutput.storage().array()
    } else {
      gradOutput.contiguous().storage().array()
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