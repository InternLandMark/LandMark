/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   gpu_matrix.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Matrix whose data resides in GPU (CUDA) memory
 */

#pragma once


#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>
#include "common.h"

enum class MatrixLayout {
	RowMajor = 0,
	SoA = 0, // For data matrices TCNN's convention is RowMajor == SoA (struct of arrays)
	ColumnMajor = 1,
	AoS = 1,
};

static constexpr MatrixLayout RM = MatrixLayout::RowMajor;
static constexpr MatrixLayout SoA = MatrixLayout::SoA;
static constexpr MatrixLayout CM = MatrixLayout::ColumnMajor;
static constexpr MatrixLayout AoS = MatrixLayout::AoS;


template <typename T>
struct MatrixView {
	__host__ __device__ MatrixView() : data{nullptr}, stride_i{0}, stride_j{0} {}
	__host__ __device__ MatrixView(T* data, uint32_t stride_i, uint32_t stride_j) : data{data}, stride_i{stride_i}, stride_j{stride_j} {}

	__host__ __device__ MatrixView(const MatrixView<std::remove_const_t<T>>& other) : data{other.data}, stride_i{other.stride_i}, stride_j{other.stride_j} {}

	__host__ __device__ T& operator()(uint32_t i, uint32_t j = 0) const {
		return data[i * stride_i + j * stride_j];
	}

	__host__ __device__ void advance(uint32_t m, uint32_t n) {
		data = &(*this)(m, n);
	}

	__host__ __device__ void advance_rows(uint32_t m) {
		advance(m, 0);
	}

	__host__ __device__ void advance_cols(uint32_t n) {
		advance(0, n);
	}

	__host__ __device__ explicit operator bool() const {
		return data;
	}

	T* data;
	uint32_t stride_i, stride_j;
};
