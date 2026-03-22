#ifndef RUN_ACTION_HH
#define RUN_ACTION_HH

#include "G4UserRunAction.hh"
#include "G4String.hh"

class G4GenericMessenger;

class RunAction : public G4UserRunAction
{
public:
    RunAction();
    ~RunAction() override;

    void BeginOfRunAction(const G4Run* run) override;
    void EndOfRunAction(const G4Run* run) override;

    void SetOutputFileName(const G4String& name) { fOutputFileName = name; }

private:
    void DefineCommands();

    G4String fOutputFileName;
    G4GenericMessenger* fMessenger;
};

#endif
