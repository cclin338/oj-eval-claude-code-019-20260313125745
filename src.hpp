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
    // Keys/values are in HBM and should stay there

    // Step 1: Build K_all by concatenating keys in HBM
    Matrix* K_all = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      if (j == 0) {
        // Copy first key
        K_all = matrix_memory_allocator.Allocate("K_all");
        gpu_sim.Copy(keys[j], K_all, kInGpuHbm);
      } else {
        // Concatenate in HBM
        Matrix* K_new = matrix_memory_allocator.Allocate("K_new");
        gpu_sim.Concat(K_all, keys[j], K_new, 0, kInGpuHbm);
        // Release old K_all
        gpu_sim.ReleaseMatrix(K_all);
        K_all = K_new;
      }
    }

    // Step 2: Build V_all by concatenating values in HBM
    Matrix* V_all = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      if (j == 0) {
        // Copy first value
        V_all = matrix_memory_allocator.Allocate("V_all");
        gpu_sim.Copy(values[j], V_all, kInGpuHbm);
      } else {
        // Concatenate in HBM
        Matrix* V_new = matrix_memory_allocator.Allocate("V_new");
        gpu_sim.Concat(V_all, values[j], V_new, 0, kInGpuHbm);
        // Release old V_all
        gpu_sim.ReleaseMatrix(V_all);
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
    // Apply exp element-wise first
    Matrix* QK_exp = matrix_memory_allocator.Allocate("QK_exp");
    gpu_sim.MatExp(QK, QK_exp);

    // Release QK to free memory
    gpu_sim.ReleaseMatrix(QK);

    // For each row, compute sum and divide
    // Note: We need to process each row separately for proper softmax
    Matrix* QK_softmax = nullptr;
    for (size_t row = 0; row <= i; ++row) {
      // Get row from QK_exp
      Matrix* row_exp = matrix_memory_allocator.Allocate("row_exp");
      gpu_sim.GetRow(QK_exp, row, row_exp, kInSharedMemory);

      // Sum the row
      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_exp, row_sum);

      // Divide row by sum
      Matrix* row_softmax = matrix_memory_allocator.Allocate("row_softmax");
      gpu_sim.MatDiv(row_exp, row_sum, row_softmax);

      // Concatenate rows
      if (row == 0) {
        QK_softmax = row_softmax;
      } else {
        Matrix* new_rows = matrix_memory_allocator.Allocate("new_rows");
        gpu_sim.Concat(QK_softmax, row_softmax, new_rows, 0, kInSharedMemory);
        // Release old QK_softmax
        if (row > 1) {
          gpu_sim.ReleaseMatrix(QK_softmax);
        }
        QK_softmax = new_rows;
        // Release row_softmax as it's now part of new_rows
        gpu_sim.ReleaseMatrix(row_softmax);
      }

      // Release temporary matrices
      gpu_sim.ReleaseMatrix(row_exp);
      gpu_sim.ReleaseMatrix(row_sum);
    }

    // Release QK_exp after processing all rows
    gpu_sim.ReleaseMatrix(QK_exp);

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