/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_LITE_SRC_COMMON_CRYPTO_H_
#define MINDSPORE_LITE_SRC_COMMON_CRYPTO_H_

#include <string>
#include <memory>

typedef unsigned char Byte;
namespace mindspore::lite {
constexpr size_t MAX_DEC_THREAD_NUM = 64;            // maximum number of threads can launch during dec
constexpr size_t MAX_BLOCK_SIZE = 64 * 1024 * 1024;  // Maximum ciphertext segment, units is Byte
constexpr size_t RESERVED_BYTE_PER_BLOCK = 50;       // Reserved byte per block to save addition info
constexpr size_t DECRYPT_BLOCK_BUF_SIZE = MAX_BLOCK_SIZE + RESERVED_BYTE_PER_BLOCK;  // maximum length of decrypt block
constexpr unsigned int GCM_MAGIC_NUM = 0x7F3A5ED8;                                   // Magic number
constexpr unsigned int CBC_MAGIC_NUM = 0x7F3A5ED9;                                   // Magic number
constexpr unsigned int SM4_CBC_MAGIC_NUM = 0x7F3A5EDA;                               // Magic number
constexpr size_t Byte16 = 16;
constexpr size_t RAND_SEED_LENGTH = 48;
constexpr char kRandomPath[] = "/dev/random";

std::unique_ptr<Byte[]> Encrypt(size_t *encrypt_len, const Byte *plain_data, size_t plain_len, const Byte *key,
                                size_t key_len, const std::string &enc_mode);
std::unique_ptr<Byte[]> Decrypt(size_t *decrypt_len, const std::string &encrypt_data_path, const Byte *key,
                                size_t key_len, const std::string &dec_mode);
std::unique_ptr<Byte[]> Decrypt(size_t *decrypt_len, const Byte *model_data, size_t data_size, const Byte *key,
                                size_t key_len, const std::string &dec_mode);
std::unique_ptr<Byte[]> Decrypt(size_t *decrypt_len, const std::string &encrypt_data_path, const Byte *key,
                                size_t key_len, const std::string &dec_mode, size_t num_threads);
}  // namespace mindspore::lite
#endif
