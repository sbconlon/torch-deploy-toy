// torch includes
#include <torch/script.h>

// tbb includes
#include <tbb/tbb.h>

// std lib includes
#include <iostream>
#include <vector>


int main() {
    // deserialize ScriptModule
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("../scripted_mnist.pt");
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model\n";
        //std::cerr << e.msg_without_backtrace();
        return -1;
    }
    std::cout << "Model loaded successfully\n";
    
    torch::NoGradGuard no_grad; // ensures that autograd is off
    module.eval();              // turn off dropout and other training-time layers/functions
    // Parallel inference loop
    tbb::task_scheduler_init init(10); // initialize 10 threads
    tbb::mutex printMutex;
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, 100), // process 100 images
        [&](const tbb::blocked_range<size_t>& r) {
            for(size_t img_num = r.begin(); img_num != r.end(); ++img_num) {
                // Print info to confirm parallelization
                printMutex.lock();
                std::cout << "Thread " << tbb::this_tbb_thread::get_id() << " is processing image " << img_num << std::endl;
                printMutex.unlock();
                // create an input "image"
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(torch::rand({1, 784}));
                // execute model and package output as a tensor
                at::Tensor output = module.forward(inputs).toTensor();
            }
        }
    );
        
    
    
    std::cout << "\nDONE\n";
    return 0;

}
