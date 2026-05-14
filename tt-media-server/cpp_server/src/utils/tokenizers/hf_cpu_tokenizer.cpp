// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/tokenizers/hf_cpu_tokenizer.hpp"

#include <cstdlib>
#include <sstream>

namespace tt::utils::tokenizers {

std::string HFCPUTokenizer::modelName() const {
  // Return the model name from env or a generic default
  const char* model = std::getenv("HF_MODEL");
  if (model && *model) {
    return model;
  }
  return "hf-cpu-model";
}

std::vector<int64_t> HFCPUTokenizer::stopTokenIds() const {
  // Read stop token IDs from env HF_STOP_TOKEN_IDS
  const char* env = std::getenv("HF_STOP_TOKEN_IDS");
  if (env && *env) {
    std::vector<int64_t> ids;
    std::string s(env);
    size_t pos = 0;
    while (pos < s.size()) {
      size_t comma = s.find(',', pos);
      std::string token = s.substr(pos, comma - pos);
      // Trim whitespace
      size_t start = token.find_first_not_of(" \t");
      size_t end = token.find_last_not_of(" \t");
      if (start != std::string::npos && end != std::string::npos) {
        ids.push_back(std::stoll(token.substr(start, end - start + 1)));
      }
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
    if (!ids.empty()) {
      return ids;
    }
  }
  // Fallback: use EOS token from config
  if (!cfg_.eos_token.empty()) {
    // We don't have the token ID here, just the string.
    // The Python runner handles stop tokens, so return empty.
  }
  return {};
}

std::string HFCPUTokenizer::applyChatTemplate(
    const std::vector<tt::domain::llm::ChatMessage>& messages,
    bool addGenerationPrompt,
    const std::optional<std::vector<tt::domain::tool_calls::Tool>>& /*tools*/,
    [[maybe_unused]] bool enableReasoning, bool skipApplyChatTemplate) const {
  if (skipApplyChatTemplate) {
    std::ostringstream out;
    for (const auto& m : messages) {
      out << m.content;
    }
    return out.str();
  }

  // Use the jinja chat template from tokenizer_config.json if available
  if (!cfg_.chat_template.empty()) {
    // TODO: Implement jinja template rendering
    // For now, fall through to simple format
  }

  // Simple fallback: use <|im_start|>/|<im_end|> format (Qwen-style)
  // or generic format if not Qwen
  std::ostringstream out;
  for (const auto& m : messages) {
    if (m.role == "system") {
      out << "<|im_start|>system\n" << m.content << "<|im_end|>\n";
    } else if (m.role == "user") {
      out << "<|im_start|>user\n" << m.content << "<|im_end|>\n";
    } else if (m.role == "assistant") {
      out << "<|im_start|>assistant\n" << m.content << "<|im_end|>\n";
    }
  }
  if (addGenerationPrompt) {
    out << "<|im_start|>assistant\n";
  }
  return out.str();
}

}  // namespace tt::utils::tokenizers
