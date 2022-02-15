#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

const int OP_MAX_SIZE = 30;
const int EPOCHS = 1;
const int GENS = 1000000;
const int BIRTHS = 200000;
const int SURVIVE_BEST = 5000;
const int SURVIVE_RAND = 50000;
const int MUTATIONS_RATIO = 100;

const double INF = 1e9;

static std::random_device rd;
static std::mt19937 rnd(rd());

class Operation {
public:
    virtual double operator()(double x, double y) const = 0;

    virtual void Print(std::ostream &out) const = 0;

    [[nodiscard]] size_t Size() const {
        return size;
    }

    [[nodiscard]] virtual std::shared_ptr<Operation> RandomNode() = 0;

    [[nodiscard]] virtual std::shared_ptr<Operation> Mutated() = 0;

    virtual ~Operation() = default;

protected:
    size_t size = 1;
};

std::ostream &operator<<(std::ostream &out, const Operation &operation) {
    operation.Print(out);
    return out;
}

class Constant : public Operation, public std::enable_shared_from_this<Constant> {
public:
    explicit Constant(double c) : c_(c) {
    }

    double operator()(double x, double y) const override {
        return c_;
    }

    void Print(std::ostream &out) const override {
        out << c_;
    }

    std::shared_ptr<Operation> RandomNode() override {
        return shared_from_this();
    }

    std::shared_ptr<Operation> Mutated() override {
        return std::make_shared<Constant>(rnd() % 2);
    }

private:
    double c_;
};

class X : public Operation, public std::enable_shared_from_this<X> {
public:
    double operator()(double x, double y) const override {
        return x;
    }

    void Print(std::ostream &out) const override {
        out << "x";
    }

    std::shared_ptr<Operation> RandomNode() override {
        return shared_from_this();
    }

    std::shared_ptr<Operation> Mutated() override {
        return std::make_shared<Constant>(rnd() % 2);
    }
};

class Y : public Operation, public std::enable_shared_from_this<Y> {
public:
    double operator()(double x, double y) const override {
        return y;
    }

    void Print(std::ostream &out) const override {
        out << "y";
    }

    std::shared_ptr<Operation> RandomNode() override {
        return shared_from_this();
    }

    std::shared_ptr<Operation> Mutated() override {
        return std::make_shared<Constant>(rnd() % 2);
    }
};

class UnaryOperation : public Operation, public std::enable_shared_from_this<UnaryOperation> {
public:
    explicit UnaryOperation(std::string func_name_, std::function<double(double)> func_,
                            std::shared_ptr<Operation> arg_)
            : func_name(std::move(func_name_)), func(std::move(func_)), arg(std::move(arg_)) {
        size = (arg ? arg->Size() : 0) + 1;
    }

    double operator()(double x, double y) const override {
        return func(arg->operator()(x, y));
    }

    void Print(std::ostream &out) const override {
        if (func_name.size() == 1) {
            out << func_name << *arg;
        } else {
            out << func_name << "(" << *arg << ")";
        }
    }

    std::shared_ptr<Operation> RandomNode() override {
        size_t num = rnd() % size;
        if (num == 0) {
            return shared_from_this();
        }
        return arg->RandomNode();
    }

    std::shared_ptr<Operation> Mutated() override {
        size_t num = rnd() % size;
        if (num == 0) {
            return std::make_shared<Constant>(rnd() % 2);
        }
        return std::make_shared<UnaryOperation>(func_name, func, arg->Mutated());
    }

protected:
    std::string func_name;
    std::function<double(double)> func;
    std::shared_ptr<Operation> arg;
};

class BinaryOperation : public Operation, public std::enable_shared_from_this<BinaryOperation> {
public:
    BinaryOperation(std::string func_name_, std::function<double(double, double)> func_,
                    std::shared_ptr<Operation> arg1_, std::shared_ptr<Operation> arg2_)
            : func_name(std::move(func_name_)), func(std::move(func_)),
              arg1(std::move(arg1_)), arg2(std::move(arg2_)) {
        size = (arg1 ? arg1->Size() : 0) + (arg2 ? arg2->Size() : 0) + 1;
    }

    double operator()(double x, double y) const override {
        return func(arg1->operator()(x, y), arg2->operator()(x, y));
    }

    void Print(std::ostream &out) const override {
        if (func_name.size() == 1) {
            out << "(" << *arg1 << " " << func_name << " " << *arg2 << ")";
        } else {
            out << func_name << "(" << *arg1 << ", " << *arg2 << ")";
        }
    }

    std::shared_ptr<Operation> RandomNode() override {
        size_t num = rnd() % size;
        if (num == 0) {
            return shared_from_this();
        }
        if (num <= arg1->Size()) {
            return arg1->RandomNode();
        } else {
            return arg2->RandomNode();
        }
    }

    std::shared_ptr<Operation> Mutated() override {
        size_t num = rnd() % size;
        if (num == 0) {
            return std::make_shared<Constant>(rnd() % 2);
        }
        if (num <= arg1->Size()) {
            return std::make_shared<BinaryOperation>(func_name, func, arg1->Mutated(), arg2);
        } else {
            return std::make_shared<BinaryOperation>(func_name, func, arg1, arg2->Mutated());
        }
    }

protected:
    std::string func_name;
    std::function<double(double, double)> func;
    std::shared_ptr<Operation> arg1, arg2;
};

double CalcError(const std::shared_ptr<Operation> &op, const std::vector<std::pair<double, double>> &input,
                 const std::vector<double> &output) {
    double res = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        double val = op->operator()(input[i].first, input[i].second);
        res += (val - output[i]) * (val - output[i]);
    }
    res = sqrt(res);
    if (std::isnan(res)) {
        return INF;
    }
    return res;
}

static const std::vector<std::pair<std::string, std::function<double(double)>>> unary_functions = {
        {"-",     [](double x) { return -x; }},
        {"sqrt",  [](double x) { return sqrt(x); }},
        {"sqrt3", [](double x) { return pow(x, 1 / 3.0); }},
        {"sin",   [](double x) { return sin(x); }},
        {"cos",   [](double x) { return cos(x); }},
        {"asin",  [](double x) { return asin(x); }},
        {"acos",  [](double x) { return acos(x); }},
};

static const std::vector<std::pair<std::string, std::function<double(double, double)>>> binary_functions = {
        {"+", [](double x, double y) { return x + y; }},
        {"*", [](double x, double y) { return x * y; }},
        {"/", [](double x, double y) { return x / y; }},
};

std::shared_ptr<Operation>
Approx(const std::vector<std::pair<double, double>> &input, const std::vector<double> &output) {
    std::shared_ptr<Operation> ans;
    double ans_error = INF;
    for (int epoch_id = 1; epoch_id <= EPOCHS; ++epoch_id) {
        std::vector<std::shared_ptr<Operation>> pool = {
                std::make_shared<X>(),
                std::make_shared<Y>(),
                std::make_shared<Constant>(0),
                std::make_shared<Constant>(1),
                std::make_shared<Constant>(2),
                std::make_shared<Constant>(3),
                std::make_shared<Constant>(4),
                std::make_shared<Constant>(M_PI),
        };

        for (int gen_id = 1; gen_id <= GENS; ++gen_id) {
            std::cout << "Generation " << gen_id << " / " << GENS << "\r";
            std::cout.flush();
            size_t n = pool.size();
            for (int birth = 0; birth < BIRTHS; ++birth) {
                size_t tp = rnd() % 2;
                if (tp == 0) {
                    size_t fun_id = rnd() % unary_functions.size();
                    size_t i = rnd() % n;
                    auto subtree = pool[i]->RandomNode();
                    while (subtree->Size() + 1 > OP_MAX_SIZE) {
                        subtree = pool[i]->RandomNode();
                    }
                    auto child = std::make_shared<UnaryOperation>(
                            unary_functions[fun_id].first, unary_functions[fun_id].second, subtree);
                    pool.push_back(rnd() % MUTATIONS_RATIO == 0 ? child->Mutated() : child);
                } else {
                    size_t fun_id = rnd() % binary_functions.size();
                    size_t i = rnd() % n;
                    size_t j = rnd() % n;
                    auto subtree1 = pool[i]->RandomNode();
                    auto subtree2 = pool[j]->RandomNode();
                    while (subtree1->Size() + subtree2->Size() + 1 > OP_MAX_SIZE) {
                        subtree1 = pool[i]->RandomNode();
                        subtree2 = pool[j]->RandomNode();
                    }
                    auto child = std::make_shared<BinaryOperation>(
                            binary_functions[fun_id].first, binary_functions[fun_id].second, subtree1, subtree2);
                    pool.push_back(rnd() % MUTATIONS_RATIO == 0 ? child->Mutated() : child);
                }
            }
            int pref = std::min((int) pool.size(), SURVIVE_BEST);
            std::sort(pool.begin(), pool.end(), [&](const auto &f1, const auto &f2) {
                return CalcError(f1, input, output) < CalcError(f2, input, output);
            });
            std::vector<std::shared_ptr<Operation>> new_pool(pref);
            std::copy(pool.begin(), pool.begin() + pref, new_pool.begin());
            for (int rand_it = 0; rand_it < SURVIVE_RAND; ++rand_it) {
                size_t i = rnd() % pool.size();
                new_pool.emplace_back(pool[i]);
            }
            pool = std::move(new_pool);
            double error = CalcError(pool[0], input, output);
            if (error < ans_error) {
                ans_error = error;
                ans = pool[0];
                std::cout << "IMPROVED gen = " << gen_id << ", error = " << error << ", ans = " << *ans << std::endl;
            }
//            for (int i = 0; i < 10; ++i) {
//                std::cout << "Place: " << i + 1 << ", error = " << CalcError(pool[i], input, output) << std::endl;
//            }
//            std::cout << std::endl;
        }
    }
    return ans;
}

const int T = 100;
const int CNT_LAYERS = 2;

int main() {
    std::vector<std::vector<double>> a1(T + 1, std::vector<double>(2 * T + 1));
    std::vector<std::vector<double>> a2(T + 1, std::vector<double>(2 * T + 1));
    a2[0][T] = 1;
    for (int t = 1; t <= T; ++t) {
        for (int x = -t; x <= t; x += 2) {
            if (x < T) {
                a1[t][T + x] = (a2[t - 1][T + x + 1] + a1[t - 1][T + x + 1]) / sqrt(2);
            }
            if (x > -T) {
                a2[t][T + x] = (a2[t - 1][T + x - 1] - a1[t - 1][T + x - 1]) / sqrt(2);
            }
        }
    }
//    std::cout.precision(2);
//    for (int t = 0; t <= T; ++t) {
//        for (int x = 0; x < 2 * T + 1; ++x) {
//            std::cout << a2[t][x] << '\t';
//        }
//        std::cout << std::endl;
//    }

    std::vector<std::pair<double, double>> input;
    std::vector<double> output;
    for (int t = T - CNT_LAYERS + 1; t <= T; ++t) {
        for (int x = -t; x <= t; x += 2) {
            if (t * t > 2 * x * x) {
                input.emplace_back(t, x);
                output.push_back(a1[t][T + x] / (sqrt(2 / M_PI) * pow(t * t - 2 * x * x, -0.25)));
                std::cout << output.back() << ' ';
            }
        }
        std::cout << std::endl;
    }

//    std::vector<std::pair<double, double>> input = {
//            {1, 3},
//            {2, 1},
//            {3, 4},
//            {4, 8}
//    };
//    std::vector<double> output = {13, 7, 37, 112};

    auto ans = Approx(input, output);
    std::cout << "Error = " << CalcError(ans, input, output) << ", ans = " << *ans << std::endl;
}
