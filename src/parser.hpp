#ifndef ESX_PARSER_HPP
#define ESX_PARSER_HPP

#include "ast.hpp"
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix.hpp>
#include <string>

/**
 * @file parser.hpp
 * @brief Parser for EasiScriptX (ESX) source code.
 * @details Uses Boost.Spirit to parse ESX syntax into AST nodes, supporting
 * tensor operations, training, LLM fine-tuning, and enterprise features.
 */

namespace esx::parser {
namespace qi = boost::spirit::qi;
namespace phoenix = boost::phoenix;

template <typename Iterator>
struct Parser : qi::grammar<Iterator, ast::Program(), qi::space_type> {
    Parser() : Parser::base_type(program) {
        using qi::double_;
        using qi::int_;
        using qi::lit;
        using qi::lexeme;
        using qi::char_;
        using qi::string;

        ident = lexeme[char_("a-zA-Z_") >> *char_("a-zA-Z0-9_")];
        tensor_lit = lit("tensor") >> '(' >> ('[' >> double_ % ',' >> ']') % ',' >> ')';
        matmul_expr = expr >> '@' >> expr;
        conv2d_expr = lit("conv2d") >> '(' >> expr >> ',' >> expr >> ',' >> lit("stride:") >> int_ >> ',' >> lit("padding:") >> int_ >> ')';
        attention_expr = lit("attention") >> '(' >> expr >> ',' >> expr >> ',' >> expr >> ',' >> lit("heads:") >> int_ >> ',' >> lit("dim:") >> int_ >> -(',' >> lit("flash=true")) >> ')';
        lora_expr = lit("lora") >> '(' >> expr >> ',' >> lit("rank:") >> int_ >> ')';
        mixed_precision_expr = lit("mixed_precision") >> '(' >> expr >> ',' >> (lit("bf16") | lit("fp16")) >> ')';
        fused_matmul_relu_expr = lit("fused_matmul_relu") >> '(' >> expr >> ',' >> expr >> ')';
        load_hf_expr = lit("load_hf") >> '(' >> lexeme[*(char_ - ')')] >> ')';
        expr = ident [qi::_val = phoenix::new_<ast::IdentExpr>(qi::_1, qi::_2)] |
               tensor_lit [qi::_val = phoenix::new_<ast::TensorLitExpr>(qi::_1, qi::_2)] |
               matmul_expr [qi::_val = phoenix::new_<ast::MatmulExpr>(qi::_1, qi::_2, qi::_3)] |
               conv2d_expr [qi::_val = phoenix::new_<ast::Conv2dExpr>(qi::_1, qi::_2, qi::_3, qi::_4, qi::_5)] |
               attention_expr [qi::_val = phoenix::new_<ast::AttentionExpr>(qi::_1, qi::_2, qi::_3, qi::_4, qi::_5, qi::_6, qi::_7)] |
               lora_expr [qi::_val = phoenix::new_<ast::LoRAExpr>(qi::_1, qi::_2, qi::_3)] |
               mixed_precision_expr [qi::_val = phoenix::new_<ast::MixedPrecisionExpr>(qi::_1, qi::_2, qi::_3)] |
               fused_matmul_relu_expr [qi::_val = phoenix::new_<ast::FusedMatmulReluExpr>(qi::_1, qi::_2, qi::_3)] |
               load_hf_expr [qi::_val = phoenix::new_<ast::LoadHFExpr>(qi::_1, qi::_2)];

        let_stmt = lit("let") >> ident >> '=' >> expr;
        train_stmt = lit("train") >> '(' >> expr >> ',' >> expr >> ',' >> lit("loss:") >> (lit("ce") | lit("mse")) >>
                     ',' >> lit("opt:") >> (lit("adam") | lit("sgd") | lit("lomo")) >> '(' >> (ident >> '=' >> double_) % ',' >> ')' >>
                     ',' >> lit("epochs:") >> int_ >> ',' >> lit("device:") >> (lit("cpu") | lit("gpu")) >> ')';
        pipeline_parallel_stmt = lit("pipeline_parallel") >> '(' >> expr >> ',' >> lit("stages:") >> int_ >> ')';
        instruction_tune_stmt = lit("instruction_tune") >> '(' >> expr >> ',' >> expr >> ',' >> lexeme[*(char_ - ')')] >> ')';
        domain_adapt_stmt = lit("adapt_domain") >> '(' >> expr >> ',' >> lexeme[*(char_ - ')')] >> ')';
        heterogeneous_schedule_stmt = lit("schedule_heterogeneous") >> '(' >> lit("cpu:") >> double_ >> ',' >> lit("gpu:") >> double_ >> ')';
        energy_aware_stmt = lit("energy_aware") >> '(' >> expr >> ',' >> lit("max_power:") >> double_ >> ')';
        switch_framework_stmt = lit("switch_framework") >> '(' >> (lit("pytorch") | lit("tensorflow") | lit("jax")) >> ',' >> lexeme[*(char_ - ')')] >> ')';
        track_experiment_stmt = lit("track_experiment") >> '(' >> lit("mlflow") >> ',' >> lexeme[*(char_ - ')')] >> ')';
        memory_broker_stmt = lit("memory_broker") >> '(' >> expr >> ',' >> lit("max_mem:") >> double_ >> ',' >> lit("strategy:") >> (lit("zeRO") | lit("offload")) >> ')';
        quantize_stmt = lit("quantize") >> '(' >> expr >> ',' >> lit("bits:") >> int_ >> ',' >> lit("method:") >> (lit("ptq") | lit("qat")) >> ')';
        speculative_decode_stmt = lit("speculative_decode") >> '(' >> expr >> ',' >> expr >> ',' >> lit("max_tokens:") >> int_ >> ')';
        pattern_recognize_stmt = lit("pattern_recognize") >> '(' >> expr >> ',' >> lit("rules:") >> (lit("geometric") | lit("arithmetic")) >> ')';
        checkpoint_stmt = lit("checkpoint") >> '(' >> expr >> ',' >> lit("segments:") >> int_ >> ')';
        agent_tune_stmt = lit("agent_tune") >> '(' >> expr >> ',' >> lit("agents:") >> int_ >> ',' >> lit("target:") >> lexeme[*(char_ - ')')] >> ')';
        visualize_stmt = lit("visualize") >> '(' >> expr >> ',' >> lit("filename:") >> lexeme[*(char_ - ')')] >> ')';
        stmt = let_stmt [qi::_val = phoenix::new_<ast::LetStmt>(qi::_1, qi::_2, qi::_3)] |
               train_stmt [qi::_val = phoenix::new_<ast::TrainStmt>(qi::_1, qi::_2, qi::_3, qi::_4, qi::_5, qi::_6, qi::_7, qi::_8)] |
               pipeline_parallel_stmt [qi::_val = phoenix::new_<ast::PipelineParallelStmt>(qi::_1, qi::_2, qi::_3)] |
               instruction_tune_stmt [qi::_val = phoenix::new_<ast::InstructionTuneStmt>(qi::_1, qi::_2, qi::_3, qi::_4)] |
               domain_adapt_stmt [qi::_val = phoenix::new_<ast::DomainAdaptStmt>(qi::_1, qi::_2, qi::_3)] |
               heterogeneous_schedule_stmt [qi::_val = phoenix::new_<ast::HeterogeneousScheduleStmt>(qi::_1, qi::_2, qi::_3)] |
               energy_aware_stmt [qi::_val = phoenix::new_<ast::EnergyAwareStmt>(qi::_1, qi::_2, qi::_3)] |
               switch_framework_stmt [qi::_val = phoenix::new_<ast::SwitchFrameworkStmt>(qi::_1, qi::_2, qi::_3)] |
               track_experiment_stmt [qi::_val = phoenix::new_<ast::TrackExperimentStmt>(qi::_1, qi::_2, qi::_3)] |
               memory_broker_stmt [qi::_val = phoenix::new_<ast::MemoryBrokerStmt>(qi::_1, qi::_2, qi::_3, qi::_4)] |
               quantize_stmt [qi::_val = phoenix::new_<ast::QuantizeStmt>(qi::_1, qi::_2, qi::_3, qi::_4)] |
               speculative_decode_stmt [qi::_val = phoenix::new_<ast::SpeculativeDecodeStmt>(qi::_1, qi::_2, qi::_3, qi::_4)] |
               pattern_recognize_stmt [qi::_val = phoenix::new_<ast::PatternRecognizeStmt>(qi::_1, qi::_2, qi::_3)] |
               checkpoint_stmt [qi::_val = phoenix::new_<ast::CheckpointStmt>(qi::_1, qi::_2, qi::_3)] |
               agent_tune_stmt [qi::_val = phoenix::new_<ast::AgentTuneStmt>(qi::_1, qi::_2, qi::_3, qi::_4)] |
               visualize_stmt [qi::_val = phoenix::new_<ast::VisualizeStmt>(qi::_1, qi::_2, qi::_3)];
        program = *stmt;
    }

private:
    qi::rule<Iterator, std::string(), qi::space_type> ident;
    qi::rule<Iterator, ast::TensorLitExpr(), qi::space_type> tensor_lit;
    qi::rule<Iterator, ast::MatmulExpr(), qi::space_type> matmul_expr;
    qi::rule<Iterator, ast::Conv2dExpr(), qi::space_type> conv2d_expr;
    qi::rule<Iterator, ast::AttentionExpr(), qi::space_type> attention_expr;
    qi::rule<Iterator, ast::LoRAExpr(), qi::space_type> lora_expr;
    qi::rule<Iterator, ast::MixedPrecisionExpr(), qi::space_type> mixed_precision_expr;
    qi::rule<Iterator, ast::FusedMatmulReluExpr(), qi::space_type> fused_matmul_relu_expr;
    qi::rule<Iterator, ast::LoadHFExpr(), qi::space_type> load_hf_expr;
    qi::rule<Iterator, std::shared_ptr<ast::Expr>(), qi::space_type> expr;
    qi::rule<Iterator, ast::LetStmt(), qi::space_type> let_stmt;
    qi::rule<Iterator, ast::TrainStmt(), qi::space_type> train_stmt;
    qi::rule<Iterator, ast::PipelineParallelStmt(), qi::space_type> pipeline_parallel_stmt;
    qi::rule<Iterator, ast::InstructionTuneStmt(), qi::space_type> instruction_tune_stmt;
    qi::rule<Iterator, ast::DomainAdaptStmt(), qi::space_type> domain_adapt_stmt;
    qi::rule<Iterator, ast::HeterogeneousScheduleStmt(), qi::space_type> heterogeneous_schedule_stmt;
    qi::rule<Iterator, ast::EnergyAwareStmt(), qi::space_type> energy_aware_stmt;
    qi::rule<Iterator, ast::SwitchFrameworkStmt(), qi::space_type> switch_framework_stmt;
    qi::rule<Iterator, ast::TrackExperimentStmt(), qi::space_type> track_experiment_stmt;
    qi::rule<Iterator, ast::MemoryBrokerStmt(), qi::space_type> memory_broker_stmt;
    qi::rule<Iterator, ast::QuantizeStmt(), qi::space_type> quantize_stmt;
    qi::rule<Iterator, ast::SpeculativeDecodeStmt(), qi::space_type> speculative_decode_stmt;
    qi::rule<Iterator, ast::PatternRecognizeStmt(), qi::space_type> pattern_recognize_stmt;
    qi::rule<Iterator, ast::CheckpointStmt(), qi::space_type> checkpoint_stmt;
    qi::rule<Iterator, ast::AgentTuneStmt(), qi::space_type> agent_tune_stmt;
    qi::rule<Iterator, ast::VisualizeStmt(), qi::space_type> visualize_stmt;
    qi::rule<Iterator, std::shared_ptr<ast::Stmt>(), qi::space_type> stmt;
    qi::rule<Iterator, ast::Program(), qi::space_type> program;
};

ast::Program parse(const std::string& input) {
    // Basic sandboxing: reject certain shell-like tokens early
    static const char* forbidden[] = {"`", "$(`", "&&", "||", "|", ";;"};
    for (auto tok : forbidden) {
        if (input.find(tok) != std::string::npos) {
            throw std::runtime_error(std::string("Forbidden token in input: ") + tok);
        }
    }
    ast::Program program;
    Parser<std::string::const_iterator> parser;
    auto iter = input.begin();
    bool r = qi::phrase_parse(iter, input.end(), parser, qi::space, program);
    if (!r || iter != input.end()) {
        int line = std::count(input.begin(), iter, '\n') + 1;
        int column = std::distance(input.rbegin(), std::find(input.rbegin(), input.rend(), '\n'));
        throw std::runtime_error("Parse error at line " + std::to_string(line) + ", column " + std::to_string(column));
    }
    program.validate();
    return program;
}

} // namespace esx::parser

#endif // ESX_PARSER_HPP