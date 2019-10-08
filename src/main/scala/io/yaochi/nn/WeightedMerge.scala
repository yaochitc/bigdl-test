package io.yaochi.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

class WeightedMerge[T: ClassTag]()
                                (implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(1 <= input.nDimension() && input.nDimension() <= 4,
      "1D, 2D, 3D or 4D tensor expected" +
        s"input dimension ${input.nDimension()}")
    val (nFrame, stride) = if (input.nDimension() == 1) {
      (1, 1)
    } else if (input.nDimension() == 2) {
      (input.size(1), 1)
    } else if (input.nDimension() == 3) {
      (1, input.size(2) * input.size(3))
    } else {
      (input.size(1), input.size(3) * input.size(4))
    }
    output.resizeAs(input)
    WeightedMerge.updateOutput[T](input, output)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(output)
    WeightedMerge.updateGradInput[T](input, gradOutput, gradInput, output)
    gradInput
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    inputShape
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