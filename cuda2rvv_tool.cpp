#include <clang/AST/AST.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Basic/TokenKinds.h>
#include <clang/Lex/Lexer.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/CommandLine.h>

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

static llvm::cl::OptionCategory ToolCategory("cuda2rvv options");

class CUDARewriter : public MatchFinder::MatchCallback {
public:
    CUDARewriter(Rewriter &R) : Rewrite(R) {}

    void run(const MatchFinder::MatchResult &Result) override {
        SourceManager &SM = Rewrite.getSourceMgr();

        // Remove CUDA kernel qualifiers (__global__, __device__, __host__)
        if (const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>("cudaFunc")) {
            SourceLocation FuncStart = FD->getSourceRange().getBegin();
            auto &LangOpts = Rewrite.getLangOpts();

            // Scan tokens at function start to remove CUDA qualifiers
            // This is a heuristic: remove "__global__", "__device__", "__host__" keywords before function name
            Token Tok;
            SourceLocation Loc = FuncStart;
            for (int i = 0; i < 10; ++i) {  // max 10 tokens to search
                bool Invalid = false;
                Loc = Lexer::GetBeginningOfToken(Loc, SM, LangOpts);
                if (Lexer::getRawToken(Loc, Tok, SM, LangOpts, true)) break;

                if (Tok.is(tok::identifier)) {
                    StringRef Spelling = Lexer::getSourceText(CharSourceRange::getTokenRange(Tok.getLocation(), Tok.getEndLoc()), SM, LangOpts);
                    if (Spelling == "__global__" || Spelling == "__device__" || Spelling == "__host__") {
                        // Remove token text including any trailing whitespace
                        SourceLocation End = Tok.getEndLoc().getLocWithOffset(1);
                        Rewrite.RemoveText(CharSourceRange::getCharRange(Tok.getLocation(), End));
                        Loc = End;
                        continue;
                    }
                }
                Loc = Tok.getEndLoc().getLocWithOffset(1);
            }
        }

        // Replace threadIdx.x, blockIdx.x, blockDim.x, gridDim.x with macros e.g. threadIdx_x
        if (const MemberExpr *ME = Result.Nodes.getNodeAs<MemberExpr>("cudaBuiltin")) {
            SourceLocation Start = ME->getSourceRange().getBegin();
            SourceLocation End = ME->getSourceRange().getEnd();
            auto &LangOpts = Rewrite.getLangOpts();

            // Extract the full text "threadIdx.x"
            StringRef Text = Lexer::getSourceText(CharSourceRange::getCharRange(Start, End.getLocWithOffset(1)), SM, LangOpts);
            // Map CUDA builtin to RVV macro form
            // "threadIdx.x" -> "threadIdx_x"
            std::string Replacement = Text.str();
            Replacement.erase(std::remove(Replacement.begin(), Replacement.end(), '.'), Replacement.end());
            Replacement.insert(Replacement.find('x'), "_");

            Rewrite.ReplaceText(SourceRange(Start, End), Replacement);
        }

        // Replace __syncthreads() with __syncthreads() macro or RVV barrier intrinsic
        if (const CallExpr *CE = Result.Nodes.getNodeAs<CallExpr>("syncthreadsCall")) {
            SourceLocation Start = CE->getCallee()->getBeginLoc();
            SourceLocation End = CE->getEndLoc();

            // Replace with __syncthreads() macro (defined in your cuda2rvv.h)
            Rewrite.ReplaceText(SourceRange(Start, End), "__syncthreads()");
        }
    }

private:
    Rewriter &Rewrite;
};

int main(int argc, const char **argv) {
    CommonOptionsParser OptionsParser(argc, argv, ToolCategory);
    ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

    class ASTConsumerImpl : public ASTConsumer {
    public:
        ASTConsumerImpl(Rewriter &R) : Handler(R) {
            // Match CUDA kernel qualifiers on functions
            Matcher.addMatcher(functionDecl(
                                   anyOf(hasAttr(clang::attr::CUDADevice),
                                         hasAttr(clang::attr::CUDAGlobal),
                                         hasAttr(clang::attr::CUDAHost)))
                                   .bind("cudaFunc"),
                               &Handler);

            // Match CUDA builtin member expressions (threadIdx.x, blockIdx.x, blockDim.x, gridDim.x)
            Matcher.addMatcher(memberExpr(
                                   member(hasAnyName("x", "y", "z")),
                                   hasObjectExpression(declRefExpr(to(varDecl(hasAnyName("threadIdx", "blockIdx", "blockDim", "gridDim")))))
                               ).bind("cudaBuiltin"),
                               &Handler);

            // Match __syncthreads() call
            Matcher.addMatcher(callExpr(callee(functionDecl(hasName("__syncthreads")))).bind("syncthreadsCall"),
                               &Handler);
        }

        void HandleTranslationUnit(ASTContext &Context) override {
            Matcher.matchAST(Context);
        }

    private:
        CUDARewriter Handler;
        MatchFinder Matcher;
    };

    class FrontendActionImpl : public ASTFrontendAction {
    public:
        void EndSourceFileAction() override {
            SourceManager &SM = TheRewriter.getSourceMgr();
            llvm::outs() << "** Transformed Source: " << SM.getFileEntryForID(SM.getMainFileID())->getName() << " **\n";
            TheRewriter.getEditBuffer(SM.getMainFileID()).write(llvm::outs());
        }

        std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
            TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
            return std::make_unique<ASTConsumerImpl>(TheRewriter);
        }

    private:
        Rewriter TheRewriter;
    };

    return Tool.run(newFrontendActionFactory<FrontendActionImpl>().get());
}
