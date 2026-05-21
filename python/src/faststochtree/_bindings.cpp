#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "faststochtree/sampler.hpp"

namespace py = pybind11;

// Convert a 2-D numpy array (any float dtype, any layout) to a contiguous
// float32 row-major buffer.  Returns the converted array (keeps it alive)
// and a const float* pointer into it.
static std::pair<py::array_t<float>, const float*>
to_float32_rowmajor(py::array arr) {
    auto out = py::array_t<float, py::array::c_style | py::array::forcecast>(arr);
    return {out, out.data()};
}

PYBIND11_MODULE(_faststochtree, m) {
    m.doc() = "faststochtree — fast BART/XBART on Apple Silicon";

    // ── BARTConfig ────────────────────────────────────────────────────────────
    py::class_<bart::BARTConfig>(m, "BARTConfig")
        .def(py::init<>())
        .def_readwrite("num_trees",        &bart::BARTConfig::num_trees)
        .def_readwrite("tree_depth",       &bart::BARTConfig::tree_depth)
        .def_readwrite("min_samples_leaf", &bart::BARTConfig::min_samples_leaf)
        .def_readwrite("alpha",            &bart::BARTConfig::alpha)
        .def_readwrite("beta",             &bart::BARTConfig::beta)
        .def_readwrite("leaf_prior_var",   &bart::BARTConfig::leaf_prior_var)
        .def_readwrite("sigma2_shape",     &bart::BARTConfig::sigma2_shape)
        .def_readwrite("sigma2_scale",     &bart::BARTConfig::sigma2_scale)
        .def_readwrite("p_eval",           &bart::BARTConfig::p_eval)
        .def_readwrite("num_threads",      &bart::BARTConfig::num_threads);

    // ── BARTModel ─────────────────────────────────────────────────────────────
    py::class_<bart::BARTModel>(m, "BARTModel")
        .def_readonly("n_samples",      &bart::BARTModel::n_samples)
        .def_readonly("n_test",         &bart::BARTModel::n_test)
        // sigma2_samples as a 1-D numpy array
        .def_property_readonly("sigma2_samples", [](const bart::BARTModel& self) {
            return py::array_t<float>(
                {(py::ssize_t)self.n_samples},
                self.sigma2_samples.data());
        })
        // test_samples as a 2-D [n_samples × n_test] numpy array
        .def_property_readonly("test_samples", [](const bart::BARTModel& self) {
            if (self.n_test == 0)
                return py::array_t<float>(std::vector<py::ssize_t>{0, 0});
            return py::array_t<float>(
                {(py::ssize_t)self.n_samples, (py::ssize_t)self.n_test},
                self.test_samples.data());
        })
        // predict(X_new) → 2-D [n_samples × n_new] numpy array
        .def("predict", [](const bart::BARTModel& self, py::array X_new_raw) {
            auto [X_arr, X_ptr] = to_float32_rowmajor(X_new_raw);
            auto buf = X_arr.request();
            if (buf.ndim != 2)
                throw std::invalid_argument("X_new must be a 2-D array");
            int n_new = static_cast<int>(buf.shape[0]);
            std::vector<float> out = self.predict(X_ptr, n_new);
            return py::array_t<float>(
                {(py::ssize_t)self.n_samples, (py::ssize_t)n_new},
                out.data());
        }, py::arg("X_new"));

    // ── fit_bart ──────────────────────────────────────────────────────────────
    m.def("fit_bart",
        [](py::array X_raw, py::array_t<float> y_raw,
           py::array X_test_raw,
           int n_burnin, int n_samples, int seed,
           const bart::BARTConfig& cfg) {

            auto [X_arr, X_ptr]     = to_float32_rowmajor(X_raw);
            auto [Xt_arr, Xt_ptr]   = to_float32_rowmajor(X_test_raw);
            auto y_arr = py::array_t<float, py::array::c_style | py::array::forcecast>(y_raw);

            auto Xbuf  = X_arr.request();
            auto Xtbuf = Xt_arr.request();
            if (Xbuf.ndim != 2)  throw std::invalid_argument("X must be 2-D");
            if (Xtbuf.ndim != 2) throw std::invalid_argument("X_test must be 2-D");

            int n = static_cast<int>(Xbuf.shape[0]);
            int p = static_cast<int>(Xbuf.shape[1]);
            int n_test = static_cast<int>(Xtbuf.shape[0]);

            return std::make_unique<bart::BARTModel>(
                bart::fit_bart(X_ptr, y_arr.data(), n, p,
                               Xt_ptr, n_test, cfg,
                               n_burnin, n_samples, seed));
        },
        py::arg("X"), py::arg("y"), py::arg("X_test"),
        py::arg("n_burnin") = 200, py::arg("n_samples") = 1000,
        py::arg("seed") = 42,
        py::arg("config") = bart::BARTConfig{});

    // ── fit_warmstart_bart ────────────────────────────────────────────────────
    m.def("fit_warmstart_bart",
        [](py::array X_raw, py::array_t<float> y_raw,
           py::array X_test_raw,
           int n_gfr_burnin, int n_mcmc_burnin, int n_samples, int seed,
           bool keep_gfr_samples,
           const bart::BARTConfig& cfg) {

            auto [X_arr, X_ptr]     = to_float32_rowmajor(X_raw);
            auto [Xt_arr, Xt_ptr]   = to_float32_rowmajor(X_test_raw);
            auto y_arr = py::array_t<float, py::array::c_style | py::array::forcecast>(y_raw);

            auto Xbuf  = X_arr.request();
            auto Xtbuf = Xt_arr.request();
            if (Xbuf.ndim != 2)  throw std::invalid_argument("X must be 2-D");
            if (Xtbuf.ndim != 2) throw std::invalid_argument("X_test must be 2-D");

            int n = static_cast<int>(Xbuf.shape[0]);
            int p = static_cast<int>(Xbuf.shape[1]);
            int n_test = static_cast<int>(Xtbuf.shape[0]);

            return std::make_unique<bart::BARTModel>(
                bart::fit_warmstart_bart(X_ptr, y_arr.data(), n, p,
                                          Xt_ptr, n_test, cfg,
                                          n_gfr_burnin, n_mcmc_burnin, n_samples,
                                          seed, keep_gfr_samples));
        },
        py::arg("X"), py::arg("y"), py::arg("X_test"),
        py::arg("n_gfr_burnin") = 20, py::arg("n_mcmc_burnin") = 0,
        py::arg("n_samples") = 200,
        py::arg("seed") = 42,
        py::arg("keep_gfr_samples") = false,
        py::arg("config") = bart::BARTConfig{});

    // ── fit_xbart ─────────────────────────────────────────────────────────────
    m.def("fit_xbart",
        [](py::array X_raw, py::array_t<float> y_raw,
           py::array X_test_raw,
           int n_burnin, int n_samples, int seed,
           const bart::BARTConfig& cfg) {

            auto [X_arr, X_ptr]     = to_float32_rowmajor(X_raw);
            auto [Xt_arr, Xt_ptr]   = to_float32_rowmajor(X_test_raw);
            auto y_arr = py::array_t<float, py::array::c_style | py::array::forcecast>(y_raw);

            auto Xbuf  = X_arr.request();
            auto Xtbuf = Xt_arr.request();
            if (Xbuf.ndim != 2)  throw std::invalid_argument("X must be 2-D");
            if (Xtbuf.ndim != 2) throw std::invalid_argument("X_test must be 2-D");

            int n = static_cast<int>(Xbuf.shape[0]);
            int p = static_cast<int>(Xbuf.shape[1]);
            int n_test = static_cast<int>(Xtbuf.shape[0]);

            return std::make_unique<bart::BARTModel>(
                bart::fit_xbart(X_ptr, y_arr.data(), n, p,
                                Xt_ptr, n_test, cfg,
                                n_burnin, n_samples, seed));
        },
        py::arg("X"), py::arg("y"), py::arg("X_test"),
        py::arg("n_burnin") = 15, py::arg("n_samples") = 25,
        py::arg("seed") = 42,
        py::arg("config") = bart::BARTConfig{});

    // ── BCFConfig ─────────────────────────────────────────────────────────────
    py::class_<bart::BCFConfig>(m, "BCFConfig")
        .def(py::init<>())
        .def_readwrite("mu_config",  &bart::BCFConfig::mu_config)
        .def_readwrite("tau_config", &bart::BCFConfig::tau_config);

    // ── BCFModel ──────────────────────────────────────────────────────────────
    py::class_<bart::BCFModel>(m, "BCFModel")
        .def_readonly("n_samples", &bart::BCFModel::n_samples)
        .def_readonly("n_test",    &bart::BCFModel::n_test)
        .def_property_readonly("sigma2_samples", [](const bart::BCFModel& self) {
            return py::array_t<float>(
                {(py::ssize_t)self.n_samples},
                self.sigma2_samples.data());
        })
        .def_property_readonly("test_tau", [](const bart::BCFModel& self) {
            if (self.n_test == 0)
                return py::array_t<float>(std::vector<py::ssize_t>{0, 0});
            return py::array_t<float>(
                {(py::ssize_t)self.n_samples, (py::ssize_t)self.n_test},
                self.test_tau.data());
        })
        .def_property_readonly("test_mu", [](const bart::BCFModel& self) {
            if (self.n_test == 0)
                return py::array_t<float>(std::vector<py::ssize_t>{0, 0});
            return py::array_t<float>(
                {(py::ssize_t)self.n_samples, (py::ssize_t)self.n_test},
                self.test_mu.data());
        })
        .def("predict_tau", [](const bart::BCFModel& self, py::array X_new_raw) {
            auto [X_arr, X_ptr] = to_float32_rowmajor(X_new_raw);
            auto buf = X_arr.request();
            if (buf.ndim != 2) throw std::invalid_argument("X_new must be 2-D");
            int n_new = static_cast<int>(buf.shape[0]);
            std::vector<float> out = self.predict_tau(X_ptr, n_new);
            return py::array_t<float>(
                {(py::ssize_t)self.n_samples, (py::ssize_t)n_new}, out.data());
        }, py::arg("X_new"))
        .def("predict_mu", [](const bart::BCFModel& self, py::array X_new_mu_raw) {
            auto [X_arr, X_ptr] = to_float32_rowmajor(X_new_mu_raw);
            auto buf = X_arr.request();
            if (buf.ndim != 2) throw std::invalid_argument("X_new_mu must be 2-D");
            int n_new = static_cast<int>(buf.shape[0]);
            std::vector<float> out = self.predict_mu(X_ptr, n_new);
            return py::array_t<float>(
                {(py::ssize_t)self.n_samples, (py::ssize_t)n_new}, out.data());
        }, py::arg("X_new_mu"));

    // ── fit_bcf ───────────────────────────────────────────────────────────────
    m.def("fit_bcf",
        [](py::array X_raw, py::array_t<float> y_raw,
           py::array_t<float> z_raw, py::array_t<float> pi_hat_raw,
           py::array X_test_raw, py::array_t<float> pi_hat_test_raw,
           int n_burnin, int n_samples, int seed,
           const bart::BCFConfig& cfg) {

            auto [X_arr,  X_ptr]  = to_float32_rowmajor(X_raw);
            auto [Xt_arr, Xt_ptr] = to_float32_rowmajor(X_test_raw);
            auto y_arr       = py::array_t<float, py::array::c_style | py::array::forcecast>(y_raw);
            auto z_arr       = py::array_t<float, py::array::c_style | py::array::forcecast>(z_raw);
            auto pi_arr      = py::array_t<float, py::array::c_style | py::array::forcecast>(pi_hat_raw);
            auto pi_test_arr = py::array_t<float, py::array::c_style | py::array::forcecast>(pi_hat_test_raw);

            auto Xbuf  = X_arr.request();
            auto Xtbuf = Xt_arr.request();
            if (Xbuf.ndim != 2)  throw std::invalid_argument("X must be 2-D");
            if (Xtbuf.ndim != 2) throw std::invalid_argument("X_test must be 2-D");

            int n      = static_cast<int>(Xbuf.shape[0]);
            int p      = static_cast<int>(Xbuf.shape[1]);
            int n_test = static_cast<int>(Xtbuf.shape[0]);

            return std::make_unique<bart::BCFModel>(
                bart::fit_bcf(X_ptr, y_arr.data(), z_arr.data(), pi_arr.data(), n, p,
                              Xt_ptr, pi_test_arr.data(), n_test,
                              cfg, n_burnin, n_samples, seed));
        },
        py::arg("X"), py::arg("y"), py::arg("z"), py::arg("pi_hat"),
        py::arg("X_test"), py::arg("pi_hat_test"),
        py::arg("n_burnin") = 200, py::arg("n_samples") = 1000,
        py::arg("seed") = 42,
        py::arg("config") = bart::BCFConfig{});

    // ── fit_xbcf ──────────────────────────────────────────────────────────────
    m.def("fit_xbcf",
        [](py::array X_raw, py::array_t<float> y_raw,
           py::array_t<float> z_raw, py::array_t<float> pi_hat_raw,
           py::array X_test_raw, py::array_t<float> pi_hat_test_raw,
           int n_burnin, int n_samples, int seed,
           const bart::BCFConfig& cfg) {

            auto [X_arr,  X_ptr]  = to_float32_rowmajor(X_raw);
            auto [Xt_arr, Xt_ptr] = to_float32_rowmajor(X_test_raw);
            auto y_arr       = py::array_t<float, py::array::c_style | py::array::forcecast>(y_raw);
            auto z_arr       = py::array_t<float, py::array::c_style | py::array::forcecast>(z_raw);
            auto pi_arr      = py::array_t<float, py::array::c_style | py::array::forcecast>(pi_hat_raw);
            auto pi_test_arr = py::array_t<float, py::array::c_style | py::array::forcecast>(pi_hat_test_raw);

            auto Xbuf  = X_arr.request();
            auto Xtbuf = Xt_arr.request();
            if (Xbuf.ndim != 2)  throw std::invalid_argument("X must be 2-D");
            if (Xtbuf.ndim != 2) throw std::invalid_argument("X_test must be 2-D");

            int n      = static_cast<int>(Xbuf.shape[0]);
            int p      = static_cast<int>(Xbuf.shape[1]);
            int n_test = static_cast<int>(Xtbuf.shape[0]);

            return std::make_unique<bart::BCFModel>(
                bart::fit_xbcf(X_ptr, y_arr.data(), z_arr.data(), pi_arr.data(), n, p,
                               Xt_ptr, pi_test_arr.data(), n_test,
                               cfg, n_burnin, n_samples, seed));
        },
        py::arg("X"), py::arg("y"), py::arg("z"), py::arg("pi_hat"),
        py::arg("X_test"), py::arg("pi_hat_test"),
        py::arg("n_burnin") = 15, py::arg("n_samples") = 25,
        py::arg("seed") = 42,
        py::arg("config") = bart::BCFConfig{});

    // ── fit_warmstart_bcf ─────────────────────────────────────────────────────
    m.def("fit_warmstart_bcf",
        [](py::array X_raw, py::array_t<float> y_raw,
           py::array_t<float> z_raw, py::array_t<float> pi_hat_raw,
           py::array X_test_raw, py::array_t<float> pi_hat_test_raw,
           int n_gfr_burnin, int n_mcmc_burnin, int n_samples, int seed,
           bool keep_gfr_samples,
           const bart::BCFConfig& cfg) {

            auto [X_arr,  X_ptr]  = to_float32_rowmajor(X_raw);
            auto [Xt_arr, Xt_ptr] = to_float32_rowmajor(X_test_raw);
            auto y_arr       = py::array_t<float, py::array::c_style | py::array::forcecast>(y_raw);
            auto z_arr       = py::array_t<float, py::array::c_style | py::array::forcecast>(z_raw);
            auto pi_arr      = py::array_t<float, py::array::c_style | py::array::forcecast>(pi_hat_raw);
            auto pi_test_arr = py::array_t<float, py::array::c_style | py::array::forcecast>(pi_hat_test_raw);

            auto Xbuf  = X_arr.request();
            auto Xtbuf = Xt_arr.request();
            if (Xbuf.ndim != 2)  throw std::invalid_argument("X must be 2-D");
            if (Xtbuf.ndim != 2) throw std::invalid_argument("X_test must be 2-D");

            int n      = static_cast<int>(Xbuf.shape[0]);
            int p      = static_cast<int>(Xbuf.shape[1]);
            int n_test = static_cast<int>(Xtbuf.shape[0]);

            return std::make_unique<bart::BCFModel>(
                bart::fit_warmstart_bcf(X_ptr, y_arr.data(), z_arr.data(), pi_arr.data(), n, p,
                                        Xt_ptr, pi_test_arr.data(), n_test,
                                        cfg, n_gfr_burnin, n_mcmc_burnin, n_samples,
                                        seed, keep_gfr_samples));
        },
        py::arg("X"), py::arg("y"), py::arg("z"), py::arg("pi_hat"),
        py::arg("X_test"), py::arg("pi_hat_test"),
        py::arg("n_gfr_burnin") = 20, py::arg("n_mcmc_burnin") = 0,
        py::arg("n_samples") = 200,
        py::arg("seed") = 42,
        py::arg("keep_gfr_samples") = false,
        py::arg("config") = bart::BCFConfig{});
}
