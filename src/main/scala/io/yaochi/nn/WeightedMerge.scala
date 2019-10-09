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
    WeightedMerge.updateOutput[T](input, output)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(output)
    WeightedMerge.updateGradInput[T](input, gradOutput, gradInput, output)
    gradInput
  }


  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
  }
}

object WeightedMerge {
  private def updateOutput[T: ClassTag](input: Tensor[T], output: Tensor[T])
                                       (implicit ev: TensorNumeric[T]): Tensor[T] = {
    null
  }

  def updateGradInput[T: ClassTag](input: Tensor[T], gradOutput: Tensor[T],
                                   gradInput: Tensor[T], output: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    null
  }

}