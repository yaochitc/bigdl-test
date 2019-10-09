package io.yaochi.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

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
    val scales = (1 to size).map(i => ev.exp(weight.valueAt(i)))
    val scaleSum = scales.reduceLeft((a, b) => ev.plus(a, b))

    val outputArray = output.storage().array()
    val inputArray = if (input.isContiguous()) {
      input.storage().array()
    } else {
      input.contiguous().storage().array()
    }
    val storageOffset = input.storageOffset() - 1

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.size().deep == gradOutput.size().deep,
      "input should have the same size with gradOutput" +
        s"inputsize ${input.size().deep} gradOutput ${gradOutput.size().deep}")

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

    gradInput.resizeAs(output)
    gradInput
  }


  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    require(input.size().deep == gradOutput.size().deep,
      "input should have the same size with gradOutput" +
        s"inputsize ${input.size().deep} gradOutput ${gradOutput.size().deep}")


  }
}