#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    /*
     * Implement your calculation logic here.
     * You can use the GpuSimulator instance to perform matrix operations.
     * For example:
     * gpu_sim.MoveMatrixToGpuHbm(keys[i]);
     * When your need a new matrix, to avoid memory leak, you should use
     * Matrix* new_matrix =
     * matrix_memory_allocator.Allocate(YOUR_MATRIX_NAME(string, which is
     * helpful for debugging)); It can manage the memory of matrices
     * automatically.
     */

    // Round i (0-indexed): process first i+1 keys and values
    // Q shape: [i+1, d=512]
    // Need to concatenate K[0]...K[i] into shape [i+1, 512]
    // Need to concatenate V[0]...V[i] into shape [i+1, 512]

    // Step 1: Concatenate all keys
    Matrix* K_all = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      if (j == 0) {
        // First key, move to HBM and copy
        K_all = matrix_memory_allocator.Allocate("K_all");
        gpu_sim.Copy(keys[j], K_all, kInGpuHbm);
      } else {
        // Concatenate with previous keys (vertically, axis=0)
        Matrix* K_new = matrix_memory_allocator.Allocate("K_concat_" + std::to_string(j));
        gpu_sim.Concat(K_all, keys[j], K_new, 0, kInGpuHbm);
        K_all = K_new;
      }
    }

    // Step 2: Concatenate all values
    Matrix* V_all = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      if (j == 0) {
        V_all = matrix_memory_allocator.Allocate("V_all");
        gpu_sim.Copy(values[j], V_all, kInGpuHbm);
      } else {
        Matrix* V_new = matrix_memory_allocator.Allocate("V_concat_" + std::to_string(j));
        gpu_sim.Concat(V_all, values[j], V_new, 0, kInGpuHbm);
        V_all = V_new;
      }
    }

    // Step 3: Move Q, K_all, V_all to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(K_all);
    gpu_sim.MoveMatrixToSharedMem(V_all);

    // Step 4: Transpose K_all to get K_all^T (shape: [512, i+1])
    gpu_sim.Transpose(K_all, kInSharedMemory);

    // Step 5: Compute Q × K_all^T (shape: [i+1, i+1])
    Matrix* QK = matrix_memory_allocator.Allocate("QK");
    gpu_sim.MatMul(current_query, K_all, QK);

    // Step 6: Apply softmax ROW-WISE
    // For each row, compute exp and normalize
    Matrix* QK_softmax = matrix_memory_allocator.Allocate("QK_softmax");

    // Apply exp element-wise
    Matrix* QK_exp = matrix_memory_allocator.Allocate("QK_exp");
    gpu_sim.MatExp(QK, QK_exp);

    // For each row, compute sum and divide
    // Note: We need to process each row separately for proper softmax
    Matrix* result_rows = nullptr;
    for (size_t row = 0; row <= i; ++row) {
      // Get row from QK_exp
      Matrix* row_exp = matrix_memory_allocator.Allocate("row_exp_" + std::to_string(row));
      gpu_sim.GetRow(QK_exp, row, row_exp, kInSharedMemory);

      // Sum the row
      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum_" + std::to_string(row));
      gpu_sim.Sum(row_exp, row_sum);

      // Divide row by sum
      Matrix* row_softmax = matrix_memory_allocator.Allocate("row_softmax_" + std::to_string(row));
      gpu_sim.MatDiv(row_exp, row_sum, row_softmax);

      // Concatenate rows
      if (row == 0) {
        result_rows = row_softmax;
      } else {
        Matrix* new_rows = matrix_memory_allocator.Allocate("rows_" + std::to_string(row));
        gpu_sim.Concat(result_rows, row_softmax, new_rows, 0, kInSharedMemory);
        result_rows = new_rows;
      }

      gpu_sim.ReleaseMatrix(row_exp);
      gpu_sim.ReleaseMatrix(row_sum);
    }
    QK_softmax = result_rows;

    // Step 7: Compute Softmax × V_all (shape: [i+1, 512])
    Matrix* result = matrix_memory_allocator.Allocate("result");
    gpu_sim.MatMul(QK_softmax, V_all, result);

    // Step 8: Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(result);

    // Step 9: Run the simulator
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Step 10: Commit the answer
    rater.CommitAnswer(*result);
    /*********************  End of your code *********************/
  
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu