// This tells Catch to provide a main() - only do this in one cpp file.
#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "smooshable_chain.hpp"
#include "../lib/fast-cpp-csv-parser/csv.h"


namespace linearham {


// Linear algebra tests

TEST_CASE("ColVecMatCwise", "[linalg]") {
  Eigen::VectorXd b(3);
  Eigen::MatrixXd A(3,4), B(3,4), correct_B(3,4);
  A << 1, 2.9, 3,  4,
       5, 6,   7,  8,
       9, 10,  11, 12;
  b << 0, 4, 1;
  correct_B <<  0,  0,  0,  0,
                20, 24, 28, 32,
                9, 10,  11, 12;

  ColVecMatCwise(b, A, B);
  REQUIRE(B == correct_B);

  // Check that we can use matrices as lvalues and rvalues in the same expression.
  ColVecMatCwise(b, A, A);
  REQUIRE(A == correct_B);
}


TEST_CASE("RowVecMatCwise", "[linalg]") {
  Eigen::RowVectorXd b(4);
  Eigen::MatrixXd A(3,4), B(3,4), correct_B(3,4);
  A << 1, 2.9, 3,  4,
       5, 6,   7,  8,
       9, 10,  11, 12;
  b << 0, 4, 1, 10;
  correct_B <<  0, 11.6, 3, 40,
                0, 24,   7, 80,
                0, 40,  11, 120;

  RowVecMatCwise(b, A, B);
  REQUIRE(B == correct_B);
}


TEST_CASE("SubProductMatrix", "[linalg]") {
  Eigen::MatrixXd A(3,3), correct_A(3,3);
  Eigen::VectorXd e(3);
  correct_A <<  2.5, -2.5, -5,
                  1,   -1, -2,
                  1,    1,  2;
  e << 2.5, -1, 2;
  A.setConstant(999);

  SubProductMatrix(e, A);
  REQUIRE(A == correct_A);
}


TEST_CASE("VectorByIndices", "[linalg]") {
  Eigen::VectorXd b(4), correct_b(4);
  Eigen::MatrixXd A(3,4);
  Eigen::VectorXi a(4);
  correct_b <<  9,  2.9,  7,  4;
  A << 1, 2.9, 3,  4,
       5, 6,   7,  8,
       9, 10,  11, 12;
  a << 2, 0, 1, 0;

  VectorByIndices(A, a, b);
  REQUIRE(b == correct_b);
}


TEST_CASE("BinaryMax", "[linalg]") {
  Eigen::MatrixXd left_matrix(2,3);
  left_matrix <<
  0.50, 0.71, 0.13,
  0.29, 0.31, 0.37;
  Eigen::MatrixXd right_matrix(3,2);
  right_matrix <<
  0.30, 0.37,
  0.29, 0.41,
  0.11, 0.97;
  Eigen::MatrixXd C(2,2);
  Eigen::MatrixXd correct_C(2,2);
  // 0.50*0.30 0.71*0.29 0.13*0.11, 0.50*0.37 0.71*0.41 0.13*0.97
  // 0.29*0.30 0.31*0.29 0.37*0.11, 0.29*0.37 0.31*0.41 0.37*0.97
  correct_C <<
  0.71*0.29, 0.71*0.41,
  0.31*0.29, 0.37*0.97;
  Eigen::MatrixXi C_idx(2,2);
  Eigen::MatrixXi correct_C_idx(2,2);
  correct_C_idx <<
  1,1,
  1,2;
  BinaryMax(left_matrix, right_matrix, C, C_idx);
  REQUIRE(C == correct_C);
  REQUIRE(C_idx == correct_C_idx);
}


// Core tests

TEST_CASE("BuildTransition", "[core]") {
  Eigen::VectorXd landing(3);
  landing << 0.13, 0.17, 0.19;
  Eigen::VectorXd next_transition(2);
  next_transition << 0.2, 0.3;
  Eigen::MatrixXd correct_transition(3,3);
  correct_transition <<
  // Format is landing * transition * ... * fall_off
  0.13*0.8 , 0.13*0.2*0.7 , 0.13*0.2*0.3  ,
  0        , 0.17*0.7     , 0.17*0.3      ,
  0        , 0            , 0.19          ;

  Eigen::MatrixXd transition;
  transition = BuildTransition(landing, next_transition);
  REQUIRE(transition.isApprox(correct_transition));
}


TEST_CASE("BuildMatch", "[core]") {
  Eigen::VectorXd landing(3);
  landing << 0.13, 0.17, 0.19;
  Eigen::VectorXd emission(3);
  emission << 0.5, 0.71, 0.11;
  Eigen::VectorXd next_transition(2);
  next_transition << 0.2, 0.3;
  Eigen::MatrixXd correct_match(3,3);
  correct_match <<
  // Format is landing * emission * transition * ... * fall_off
  0.13*0.5*0.8    , 0.13*0.5*0.2*0.71*0.7 , 0.13*0.5*0.2*0.71*0.3*0.11 ,
  0               , 0.17*0.71*0.7         , 0.17*0.71*0.3*0.11         ,
  0               , 0                     , 0.19*0.11                  ;
  Eigen::MatrixXd match(3,3);
  Eigen::MatrixXd transition;

  transition = BuildTransition(landing, next_transition);
  BuildMatchMatrix(transition, emission, match);
  REQUIRE(match.isApprox(correct_match));
}


// Germline tests

TEST_CASE("Germline", "[germline]") {
  Eigen::VectorXd landing(3);
  landing << 0.13, 0.17, 0.19;
  Eigen::MatrixXd emission_matrix(2,3);
  emission_matrix <<
  0.5, 0.71, 0.11,
  0.29, 0.31, 0.37;
  Eigen::VectorXd next_transition(2);
  next_transition << 0.2, 0.3;

  Germline germline(landing, emission_matrix, next_transition);

  Eigen::VectorXi emission_indices(2);
  emission_indices << 1, 0;
  Eigen::VectorXd emission(2);
  Eigen::VectorXd correct_emission(2);
  correct_emission << 0.31, 0.11;
  germline.EmissionVector(emission_indices, 1, emission);
  REQUIRE(emission == correct_emission);

  Eigen::MatrixXd correct_match(2,2);
  correct_match <<
  // Format is landing * emission * transition * ... * fall_off
  0.17*0.31*0.7         , 0.17*0.31*0.3*0.11         ,
  0                     , 0.19*0.11                  ;
  Eigen::MatrixXd match(2,2);
  Eigen::MatrixXd transition;
  germline.MatchMatrix(1, emission_indices, 1, 1, match);
  REQUIRE(match.isApprox(correct_match));
}


// Smooshable tests

TEST_CASE("Smooshable", "[smooshable]") {
  Eigen::MatrixXd A(2,3);
  A <<
  0.5, 0.71, 0.13,
  0.29, 0.31, 0.37;
  Eigen::MatrixXd B(3,2);
  B <<
  0.3,  0.37,
  0.29, 0.41,
  0.11, 0.97;

  Eigen::MatrixXd correct_AB_marginal(2,2);
  correct_AB_marginal <<
  (0.50*0.3+0.71*0.29+0.13*0.11), (0.50*0.37+0.71*0.41+0.13*0.97),
  (0.29*0.3+0.31*0.29+0.37*0.11), (0.29*0.37+0.31*0.41+0.37*0.97);
  Eigen::MatrixXd correct_AB_viterbi(2,2);
  correct_AB_viterbi <<
  0.71*0.29, 0.71*0.41,
  0.31*0.29, 0.37*0.97;
  Eigen::MatrixXi correct_AB_viterbi_idx(2,2);
  correct_AB_viterbi_idx <<
  1,1,
  1,2;

  Smooshable s_A = Smooshable(A);
  Smooshable s_B = Smooshable(B);
  Smooshable s_AB;
  Eigen::MatrixXi AB_viterbi_idx;
  std::tie(s_AB, AB_viterbi_idx) = Smoosh(s_A, s_B);

  REQUIRE(s_AB.marginal() == correct_AB_marginal);
  REQUIRE(s_AB.viterbi() == correct_AB_viterbi);
  REQUIRE(AB_viterbi_idx == correct_AB_viterbi_idx);
  REQUIRE(s_AB.scaler_count() == 0);

  Eigen::MatrixXd C(2,1);
  C <<
  0.89,
  0.43;
  Smooshable s_C = Smooshable(C);
  Eigen::MatrixXd correct_ABC_viterbi(2,1);
  correct_ABC_viterbi <<
  // 0.71*0.29*0.89 > 0.71*0.41*0.43
  // 0.31*0.29*0.89 < 0.37*0.97*0.43
  0.71*0.29*0.89,
  0.37*0.97*0.43;
  // So the Viterbi indices are
  // 0
  // 1
  //
  // This 1 in the second row then gives us the corresponding column index,
  // which contains a 2. So the second path is {2,1}.

  SmooshableVector sv = {s_A, s_B, s_C};
  SmooshableChain chain = SmooshableChain(sv);
  IntVectorVector correct_viterbi_paths = {{1,0}, {2,1}};
  REQUIRE(chain.smooshed()[0].viterbi() == correct_AB_viterbi);
  REQUIRE(chain.smooshed().back().viterbi() == correct_ABC_viterbi);
  REQUIRE(chain.viterbi_paths() == correct_viterbi_paths);
}


// Ham comparison tests

TEST_CASE("Ham Comparison 1", "[ham]") {
  Eigen::VectorXd landing_a(3);
  landing_a << 1, 1, 1;
  Eigen::MatrixXd emission_matrix_a(2,3);
  emission_matrix_a <<
  0.1, 0.2, 0.3,
  0.9, 0.8, 0.7;
  Eigen::VectorXd next_transition_a(2);
  next_transition_a << 1, 0.23;
  Germline germline_a(landing_a, emission_matrix_a, next_transition_a);
  Eigen::VectorXi emission_indices_a(3);
  emission_indices_a << 0, 1, 1;
  Smooshable s_a = SmooshableGermline(germline_a, 0, emission_indices_a,  0, 1);
  Eigen::MatrixXd correct_marginal_a(1, 2);
  correct_marginal_a << 0.1*0.8*0.77, 0.1*0.8*0.23*0.7;
  REQUIRE(s_a.marginal().isApprox(correct_marginal_a));
  REQUIRE(s_a.scaler_count() == 0);

  Eigen::VectorXd landing_b(3);
  landing_b << 1, 1, 1;
  Eigen::MatrixXd emission_matrix_b(2,3);
  emission_matrix_b <<
  0.11, 0.13, 0.17,
  0.89, 0.87, 0.83;
  Eigen::VectorXd next_transition_b(2);
  next_transition_b << 1, 1;
  Germline germline_b(landing_b, emission_matrix_b, next_transition_b);
  Eigen::VectorXi emission_indices_b(3);
  emission_indices_b << 1, 0, 0;
  Smooshable s_b = SmooshableGermline(germline_b, 0, emission_indices_b, 1, 0);
  Eigen::MatrixXd correct_marginal_b(2, 1);
  correct_marginal_b <<
  0.89*0.13*0.17,
  0.13*0.17;
  REQUIRE(s_b.marginal().isApprox(correct_marginal_b));
  REQUIRE(s_b.scaler_count() == 0);

  Smooshable s_ab;
  Eigen::MatrixXi viterbi_idx_ab;
  std::tie(s_ab, viterbi_idx_ab) = Smoosh(s_a, s_b);
  Eigen::MatrixXd correct_marginal_ab(1, 1);
  correct_marginal_ab <<
  (0.1*0.8*0.77*0.89*0.13*0.17 + 0.1*0.8*0.23*0.7*0.13*0.17);
  REQUIRE(s_ab.marginal().isApprox(correct_marginal_ab));
  Eigen::MatrixXd correct_viterbi_ab(1, 1);
  correct_viterbi_ab << 0.1*0.8*0.77*0.89*0.13*0.17;
  REQUIRE(s_ab.viterbi().isApprox(correct_viterbi_ab));
  REQUIRE(s_ab.scaler_count() == 0);

  // now let's test for underflow
  landing_b.array() *= SCALE_THRESHOLD;
  germline_b = Germline(landing_b, emission_matrix_b, next_transition_b);
  s_b = SmooshableGermline(germline_b, 0, emission_indices_b, 1, 0);
  REQUIRE(s_b.marginal().isApprox(correct_marginal_b));
  REQUIRE(s_b.scaler_count() == 1);

  std::cout << "Ham test 1 marginal: " << correct_marginal_ab << std::endl;
  std::cout << "Ham test 1 viterbi: " << correct_viterbi_ab << std::endl;
}


// IO tests

TEST_CASE("YAML", "[io]") {
  Eigen::VectorXd V_landing(5);
  V_landing << 0.6666666666666666, 0, 0, 0, 0;
  Eigen::MatrixXd V_emission_matrix(4,5);
  V_emission_matrix <<
  0.79, 0.1, 0.01, 0.55, 0.125,
  0.07, 0.1, 0.01, 0.15, 0.625,
  0.07, 0.1, 0.97, 0.15, 0.125,
  0.07, 0.7, 0.01, 0.15, 0.125;
  Eigen::VectorXd V_next_transition(4);
  V_next_transition << 1, 1, 0.8, 0.5;
  double V_gene_prob = 0.07;
  double V_n_self_transition_prob = 0.33333333333333337;
  Eigen::VectorXd V_n_emission_vector(4);
  V_n_emission_vector << 0.25, 0.25, 0.25, 0.25;

  Germline correct_V_Germline(V_landing, V_emission_matrix, V_next_transition);
  YAML::Node V_root = get_yaml_root("data/V_germline_ex.yaml");
  Germline V_Germline = Germline(V_root);
  NPadding V_NPadding = NPadding(V_root);
  // V genes can't initialize NTInsertion objects.
  // NTInsertion V_NTInsertion = NTInsertion(V_root);

  REQUIRE(V_Germline.emission_matrix() == correct_V_Germline.emission_matrix());
  REQUIRE(V_Germline.transition() == correct_V_Germline.transition());
  REQUIRE(V_Germline.gene_prob() == V_gene_prob);
  REQUIRE(V_NPadding.n_self_transition_prob() == V_n_self_transition_prob);
  REQUIRE(V_NPadding.n_emission_vector() == V_n_emission_vector);

  Eigen::VectorXd D_landing(5);
  D_landing << 0.4, 0.1, 0.05, 0, 0;
  Eigen::MatrixXd D_emission_matrix(4,5);
  D_emission_matrix <<
  0.12, 0.07, 0.05, 0.55, 0.01,
  0.12, 0.07, 0.05, 0.15, 0.97,
  0.64, 0.79, 0.05, 0.15, 0.01,
  0.12, 0.07, 0.85, 0.15, 0.01;
  Eigen::VectorXd D_next_transition(4);
  D_next_transition << 0.98, 0.95, 0.6, 0.35;
  double D_gene_prob = 0.035;
  Eigen::VectorXd D_n_landing_in(4);
  D_n_landing_in << 0.1, 0.2, 0.1, 0.05;
  Eigen::MatrixXd D_n_landing_out(4,5);
  D_n_landing_out <<
  0.45, 0.125, 0.1, 0, 0,
  0.45, 0.125, 0.1, 0, 0,
  0.45, 0.125, 0.1, 0, 0,
  0.45, 0.125, 0.1, 0, 0;
  Eigen::MatrixXd D_n_emission_matrix(4,4);
  D_n_emission_matrix <<
  0.7, 0.1, 0.1, 0.1,
  0.1, 0.7, 0.1, 0.1,
  0.1, 0.1, 0.7, 0.1,
  0.1, 0.1, 0.1, 0.7;
  Eigen::MatrixXd D_n_transition(4,4);
  D_n_transition <<
  0.075, 0.175, 0.05, 0.025,
  0.075, 0.175, 0.05, 0.025,
  0.075, 0.175, 0.05, 0.025,
  0.075, 0.175, 0.05, 0.025;

  Germline correct_D_Germline(D_landing, D_emission_matrix, D_next_transition);
  YAML::Node D_root = get_yaml_root("data/D_germline_ex.yaml");
  Germline D_Germline = Germline(D_root);
  NTInsertion D_NTInsertion = NTInsertion(D_root);
  // D genes can't initialize NPadding objects.
  // NPadding D_NPadding = NPadding(D_root);

  REQUIRE(D_Germline.emission_matrix() == correct_D_Germline.emission_matrix());
  REQUIRE(D_Germline.transition() == correct_D_Germline.transition());
  REQUIRE(D_Germline.gene_prob() == D_gene_prob);
  REQUIRE(D_NTInsertion.n_landing_in() == D_n_landing_in);
  REQUIRE(D_NTInsertion.n_landing_out() == D_n_landing_out);
  REQUIRE(D_NTInsertion.n_emission_matrix() == D_n_emission_matrix);
  REQUIRE(D_NTInsertion.n_transition() == D_n_transition);

  Eigen::VectorXd J_landing(5);
  J_landing << 0.25, 0.05, 0, 0, 0;
  Eigen::MatrixXd J_emission_matrix(4,5);
  J_emission_matrix <<
  0.91, 0.1, 0.06, 0.01, 0.08,
  0.03, 0.1, 0.06, 0.97, 0.08,
  0.03, 0.1, 0.82, 0.01, 0.76,
  0.03, 0.7, 0.06, 0.01, 0.08;
  Eigen::VectorXd J_next_transition(4);
  J_next_transition << 1, 1, 1, 1;
  double J_gene_prob = 0.015;
  Eigen::VectorXd J_n_landing_in(4);
  J_n_landing_in << 0.1, 0.2, 0.2, 0.2;
  Eigen::MatrixXd J_n_landing_out(4,5);
  J_n_landing_out <<
  0.4, 0.25, 0, 0, 0,
  0.4, 0.25, 0, 0, 0,
  0.4, 0.25, 0, 0, 0,
  0.4, 0.25, 0, 0, 0;
  Eigen::MatrixXd J_n_emission_matrix(4,4);
  J_n_emission_matrix <<
  0.94, 0.02, 0.02, 0.02,
  0.02, 0.94, 0.02, 0.02,
  0.02, 0.02, 0.94, 0.02,
  0.02, 0.02, 0.02, 0.94;
  Eigen::MatrixXd J_n_transition(4,4);
  J_n_transition <<
  0.05, 0.15, 0.075, 0.075,
  0.05, 0.15, 0.075, 0.075,
  0.05, 0.15, 0.075, 0.075,
  0.05, 0.15, 0.075, 0.075;
  double J_n_self_transition_prob = 0.96;
  Eigen::VectorXd J_n_emission_vector(4);
  J_n_emission_vector << 0.25, 0.25, 0.25, 0.25;

  Germline correct_J_Germline(J_landing, J_emission_matrix, J_next_transition);
  YAML::Node J_root = get_yaml_root("data/J_germline_ex.yaml");
  Germline J_Germline = Germline(J_root);
  NTInsertion J_NTInsertion = NTInsertion(J_root);
  NPadding J_NPadding = NPadding(J_root);

  REQUIRE(J_Germline.emission_matrix() == correct_J_Germline.emission_matrix());
  REQUIRE(J_Germline.transition() == correct_J_Germline.transition());
  REQUIRE(J_Germline.gene_prob() == J_gene_prob);
  REQUIRE(J_NTInsertion.n_landing_in() == J_n_landing_in);
  REQUIRE(J_NTInsertion.n_landing_out() == J_n_landing_out);
  REQUIRE(J_NTInsertion.n_emission_matrix() == J_n_emission_matrix);
  REQUIRE(J_NTInsertion.n_transition() == J_n_transition);
  REQUIRE(J_NPadding.n_self_transition_prob() == J_n_self_transition_prob);
  REQUIRE(J_NPadding.n_emission_vector() == J_n_emission_vector);

  // Now, let's test the V/D/J germline classes.
  VGermline V_Germ(V_root);
  DGermline D_Germ(D_root);
  JGermline J_Germ(J_root);

  REQUIRE(V_Germ.emission_matrix() == correct_V_Germline.emission_matrix());
  REQUIRE(V_Germ.transition() == correct_V_Germline.transition());
  REQUIRE(V_Germ.gene_prob() == V_gene_prob);
  REQUIRE(V_Germ.n_self_transition_prob() == V_n_self_transition_prob);
  REQUIRE(V_Germ.n_emission_vector() == V_n_emission_vector);

  REQUIRE(D_Germ.emission_matrix() == correct_D_Germline.emission_matrix());
  REQUIRE(D_Germ.transition() == correct_D_Germline.transition());
  REQUIRE(D_Germ.gene_prob() == D_gene_prob);
  REQUIRE(D_Germ.n_landing_in() == D_n_landing_in);
  REQUIRE(D_Germ.n_landing_out() == D_n_landing_out);
  REQUIRE(D_Germ.n_emission_matrix() == D_n_emission_matrix);
  REQUIRE(D_Germ.n_transition() == D_n_transition);

  REQUIRE(J_Germ.emission_matrix() == correct_J_Germline.emission_matrix());
  REQUIRE(J_Germ.transition() == correct_J_Germline.transition());
  REQUIRE(J_Germ.gene_prob() == J_gene_prob);
  REQUIRE(J_Germ.n_landing_in() == J_n_landing_in);
  REQUIRE(J_Germ.n_landing_out() == J_n_landing_out);
  REQUIRE(J_Germ.n_emission_matrix() == J_n_emission_matrix);
  REQUIRE(J_Germ.n_transition() == J_n_transition);
  REQUIRE(J_Germ.n_self_transition_prob() == J_n_self_transition_prob);
  REQUIRE(J_Germ.n_emission_vector() == J_n_emission_vector);
}


// Partis CSV parsing.
TEST_CASE("CSV", "[io]") {
  io::CSVReader<3, io::trim_chars<>, io::double_quote_escape<' ','\"'> > in("data/hmm_input.csv");
  in.read_header(io::ignore_extra_column, "seqs", "boundsbounds", "relpos");
  std::string seq, boundsbounds_str, relpos_str;
  in.read_row(seq, boundsbounds_str, relpos_str);  // First line.
  std::map<std::string, int> relpos_m =
    YAML::Load(relpos_str).as<std::map<std::string, int>>();
  REQUIRE(relpos_m["IGHJ6*02"] == 333);
  REQUIRE(relpos_m["IGHD2-15*01"] == 299);
  in.read_row(seq, boundsbounds_str, relpos_str);  // Second line.
  std::string correct_seq = "CAGGTGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCCTGCAAGGCTTCTGGATACACCTTCACCGGCTACTATATGCACTGGGTGCGACAGGCCCCTGGACAAGGGCTTGAGTGGATGGGATGGATCAACCCTAACAGTGGTGGCACAAACTATGCACAGAAGTTTCAGGGCTGGGTCACCATGACCAGGGACACGTCCATCAGCACAGCCTACATGGAGCTGAGCAGGCTGAGATCTGACGACACGGCCGTGTATTACTGTGCGAGAGATTTTTTATATTGTAGTGGTGGTAGCTGCTACTCCGGGGGGACTACTACTACTACGGTATGGACGTCTGGGGGCAAGGGACCACGGTCACCGTCTCCTCA";
  REQUIRE(seq == correct_seq);
  std::map<std::string, std::pair<int, int>> bb_map =
    YAML::Load(boundsbounds_str).as<std::map<std::string, std::pair<int, int>>>();
  REQUIRE(bb_map["v_l"].second == 2);
  REQUIRE(bb_map["d_r"].first == 328);
}
}
