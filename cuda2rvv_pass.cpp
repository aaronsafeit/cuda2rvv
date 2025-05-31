// CUDA2RVVPass.cpp
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/PatternMatch.h"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

struct CUDA2RVVPass : public FunctionPass {
  static char ID;
  CUDA2RVVPass() : FunctionPass(ID) {}

  // Helper: Create RVV intrinsic call (example for vsetvl)
  CallInst *createRVVVsetvl(IRBuilder<> &Builder, Value *AVL) {
    Module *M = Builder.GetInsertBlock()->getModule();
    // This name depends on your LLVM RISCV target support
    Function *VsetvlFunc = M->getFunction("llvm.riscv.vsetvl");
    if (!VsetvlFunc) {
      // Declare the intrinsic if missing
      LLVMContext &Ctx = M->getContext();
      FunctionType *FT = FunctionType::get(
          Type::getInt32Ty(Ctx),
          {Type::getInt32Ty(Ctx)},
          false);
      VsetvlFunc = Function::Create(FT, Function::ExternalLinkage,
                                   "llvm.riscv.vsetvl", M);
    }
    return Builder.CreateCall(VsetvlFunc, AVL, "vl");
  }

  // Lower barrier intrinsic to a call to pthread barrier or fence
  void lowerBarrier(CallInst *CI) {
    IRBuilder<> Builder(CI);

    Module *M = CI->getModule();
    LLVMContext &Ctx = M->getContext();

    // Example: Replace with fence instruction
    Instruction *Fence = Builder.CreateFence(AtomicOrdering::SequentiallyConsistent,
                                            SyncScope::System);
    CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
    CI->eraseFromParent();
  }

  // Lower shfl.sync intrinsic: this is complex and often implemented via vector gather + masking
  void lowerShflSync(CallInst *CI) {
    // This is a stub to demonstrate replacement - real lowering is complex
    IRBuilder<> Builder(CI);

    // For example: Just replace with input argument (no-op)
    Value *Input = CI->getArgOperand(0);
    CI->replaceAllUsesWith(Input);
    CI->eraseFromParent();
  }

  // Lower ballot.sync intrinsic: use vector mask compress/extract
  void lowerBallotSync(CallInst *CI) {
    IRBuilder<> Builder(CI);

    // Stub: Replace with a constant true mask for demonstration
    LLVMContext &Ctx = CI->getModule()->getContext();
    Value *TrueMask = ConstantInt::get(Type::getInt32Ty(Ctx), ~0u);

    CI->replaceAllUsesWith(TrueMask);
    CI->eraseFromParent();
  }

  // Lower atomic intrinsics to RVV atomic intrinsics or inline asm
  void lowerAtomic(CallInst *CI) {
    IRBuilder<> Builder(CI);

    // Example: convert llvm.nvvm.atomic.load.add.* to llvm.riscv.atomic.vamo.add
    // This is just a demonstration stub - real lowering requires exact intrinsic knowledge.

    // Just remove for now
    CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
    CI->eraseFromParent();
  }

  // Lower warp vote intrinsics (e.g., __all, __any) to vector mask reductions
  void lowerWarpVote(CallInst *CI) {
    IRBuilder<> Builder(CI);

    // Stub: replace with argument (no-op)
    Value *Arg0 = CI->getArgOperand(0);
    CI->replaceAllUsesWith(Arg0);
    CI->eraseFromParent();
  }

  // Lower lane id retrieval: map to threadIdx_x or sim lane ID in RVV
  void lowerLaneId(CallInst *CI) {
    IRBuilder<> Builder(CI);

    // For demonstration, replace with threadIdx_x (external global variable)
    Module *M = CI->getModule();
    LLVMContext &Ctx = M->getContext();
    GlobalVariable *ThreadIdxX = M->getGlobalVariable("threadIdx_x");
    if (!ThreadIdxX) {
      ThreadIdxX = new GlobalVariable(*M, Type::getInt32Ty(Ctx), false,
                                     GlobalValue::ExternalLinkage,
                                     nullptr, "threadIdx_x");
    }
    CI->replaceAllUsesWith(Builder.CreateLoad(Type::getInt32Ty(Ctx), ThreadIdxX));
    CI->eraseFromParent();
  }

  // Sanitize IR: remove CUDA-specific keywords/metadata (stub)
  bool sanitizeCUDAKeywords(Function &F) {
    bool changed = false;
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *Call = dyn_cast<CallInst>(&I)) {
          Function *CF = Call->getCalledFunction();
          if (!CF) continue;

          StringRef FName = CF->getName();

          // Example: remove calls to cudaMalloc, cudaFree, etc.
          if (FName.startswith("cuda")) {
            // For now just warn and remove calls
            errs() << "Sanitizing CUDA runtime call: " << FName << "\n";
            Call->replaceAllUsesWith(UndefValue::get(Call->getType()));
            Call->eraseFromParent();
            changed = true;
            break; // iterator invalidated
          }
        }
      }
    }
    return changed;
  }

  bool runOnFunction(Function &F) override {
    bool changed = false;
    for (auto &BB : F) {
      for (auto it = BB.begin(), end = BB.end(); it != end;) {
        Instruction *I = &*it++;
        if (auto *CI = dyn_cast<CallInst>(I)) {
          Function *calledFunc = CI->getCalledFunction();
          if (!calledFunc)
            continue;

          StringRef fname = calledFunc->getName();

          if (fname.startswith("llvm.nvvm.barrier0")) {
            lowerBarrier(CI);
            changed = true;
          } else if (fname.startswith("llvm.nvvm.shfl.sync")) {
            lowerShflSync(CI);
            changed = true;
          } else if (fname.startswith("llvm.nvvm.ballot.sync")) {
            lowerBallotSync(CI);
            changed = true;
          } else if (fname.startswith("llvm.nvvm.atomic.load.add") ||
                     fname.startswith("llvm.nvvm.atomic.load.exch")) {
            lowerAtomic(CI);
            changed = true;
          } else if (fname.startswith("llvm.nvvm.vote.any") ||
                     fname.startswith("llvm.nvvm.vote.all")) {
            lowerWarpVote(CI);
            changed = true;
          } else if (fname == "__laneid") {
            lowerLaneId(CI);
            changed = true;
          }
        }
      }
    }

    // Optionally sanitize CUDA keywords in this function
    changed |= sanitizeCUDAKeywords(F);

    return changed;
  }
};

} // namespace

char CUDA2RVVPass::ID = 0;

static RegisterPass<CUDA2RVVPass> X("cuda2rvv",
                                   "Lower CUDA Intrinsics and Runtime Calls to RVV",
                                   false, false);
