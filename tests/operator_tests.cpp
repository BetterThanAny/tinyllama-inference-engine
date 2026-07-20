#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <vector>

#include "test_support.hpp"
#include "tlie/operators.hpp"

namespace {

using tlie::test::Context;

std::vector<float> Floats(const nlohmann::json& value) { return value.get<std::vector<float>>(); }

void CheckVector(Context& context, const std::vector<float>& actual,
                 const std::vector<float>& expected, const float tolerance) {
  TLIE_CHECK(context, actual.size() == expected.size());
  if (actual.size() != expected.size()) {
    return;
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    TLIE_CHECK_NEAR(context, actual[index], expected[index], tolerance);
  }
}

}  // namespace

int main() {
  Context context;
  std::ifstream input(std::filesystem::path(TLIE_SOURCE_DIR) / "data/golden/operators.json");
  TLIE_CHECK(context, static_cast<bool>(input));
  if (!input) {
    return context.Finish("tlie_operator_tests");
  }
  nlohmann::json golden;
  input >> golden;
  const float tolerance = golden.at("atol").get<float>();

  const auto& rms = golden.at("rms_norm");
  const auto rms_input = Floats(rms.at("input"));
  const auto rms_weight = Floats(rms.at("weight"));
  std::vector<float> rms_output(rms_input.size());
  TLIE_CHECK(context,
             tlie::RmsNorm(rms_input, rms_weight, rms.at("epsilon").get<float>(), rms_output).ok());
  CheckVector(context, rms_output, Floats(rms.at("output")), tolerance);

  const auto& rope = golden.at("rope");
  auto query = Floats(rope.at("query"));
  auto key = Floats(rope.at("key"));
  TLIE_CHECK(context,
             tlie::ApplyRope(query, key, rope.at("query_heads").get<std::size_t>(),
                             rope.at("key_value_heads").get<std::size_t>(),
                             rope.at("head_dim").get<std::size_t>(),
                             rope.at("position").get<std::size_t>(), rope.at("theta").get<float>())
                 .ok());
  CheckVector(context, query, Floats(rope.at("output_query")), tolerance);
  CheckVector(context, key, Floats(rope.at("output_key")), tolerance);

  const auto& softmax = golden.at("softmax");
  auto probabilities = Floats(softmax.at("input"));
  TLIE_CHECK(context, tlie::SoftmaxInPlace(probabilities).ok());
  CheckVector(context, probabilities, Floats(softmax.at("output")), tolerance);

  const auto& attention = golden.at("attention");
  const auto attention_query = Floats(attention.at("query"));
  const auto keys = Floats(attention.at("keys"));
  const auto values = Floats(attention.at("values"));
  std::vector<float> attention_output(attention_query.size());
  TLIE_CHECK(context,
             tlie::AttentionReference(attention_query, keys, values,
                                      attention.at("sequence_length").get<std::size_t>(),
                                      attention.at("query_heads").get<std::size_t>(),
                                      attention.at("key_value_heads").get<std::size_t>(),
                                      attention.at("head_dim").get<std::size_t>(), attention_output)
                 .ok());
  CheckVector(context, attention_output, Floats(attention.at("output")), tolerance);

  const auto& silu = golden.at("silu_multiply");
  const auto gate = Floats(silu.at("gate"));
  const auto up = Floats(silu.at("up"));
  std::vector<float> activated(gate.size());
  TLIE_CHECK(context, tlie::SiluMultiply(gate, up, activated).ok());
  CheckVector(context, activated, Floats(silu.at("output")), tolerance);

  std::vector<float> invalid = {0.0F, std::numeric_limits<float>::infinity()};
  const auto invalid_softmax = tlie::SoftmaxInPlace(invalid);
  TLIE_CHECK(context, !invalid_softmax.ok());
  TLIE_CHECK(context, invalid_softmax.error().code == tlie::ErrorCode::kNumerical);

  std::vector<float> empty;
  const auto empty_softmax = tlie::SoftmaxInPlace(empty);
  TLIE_CHECK(context, !empty_softmax.ok());
  TLIE_CHECK(context, empty_softmax.error().code == tlie::ErrorCode::kInvalidShape);
  return context.Finish("tlie_operator_tests");
}
