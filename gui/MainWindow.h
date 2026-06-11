#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProcess>
#include <QTextEdit>
#include <QLineEdit>
#include <QLabel>
#include <QTimer>
#include <QPushButton>
#include <QRadioButton>
#include <QGroupBox>
#include <QSpinBox>
#include <QComboBox>
#include "BoardWidget.h"
#include "GameSettingsDialog.h"

class QVBoxLayout;

class EvalBarWidget : public QWidget {
public:
    EvalBarWidget(QWidget* parent = nullptr);
    void setScore(int cp);
    void setNormMode(int mode); // 0=Raw, 1=+1.0, 2=+3.0
protected:
    void paintEvent(QPaintEvent* event) override;
private:
    int score = 0;
    int normMode = 0;
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void readEngineOutput();
    void sendCommand(const QString &cmd);
    void onUserMoveRequested(const QString &move);
    void newGame();
    void openSettings();
    void triggerEngineMove();
    void onTimerTick();
    
    void toggleAnalysis(bool active);
    void toggleEditor(bool active);
    void onEditorBrushChanged();
    void onBoardEdited();

private:
    QProcess *engineProcess;
    BoardWidget *boardWidget;
    EvalBarWidget *evalBar;
    QComboBox *comboEvalNorm;
    
    QTextEdit *logWindow;
    QLineEdit *cmdInput;
    
    QLabel *statusLabel;
    QLabel *scoreLabel;
    
    QLabel *wClockLabel;
    QLabel *bClockLabel;
    
    QPushButton *btnAnalysis;
    QGroupBox *editorGroup;
    QRadioButton *rbHand, *rbWhite, *rbBlack, *rbRed, *rbEmpty;
    QSpinBox *spinCapW, *spinCapB;
    QLineEdit *kfenInputEditor;

    QTimer *gameTimer;
    int wTimeMs = 300000;
    int bTimeMs = 300000;

    PlayerType wPlayer = PlayerType::Human;
    PlayerType bPlayer = PlayerType::Engine;
    EngineMode engineMode = EngineMode::FixedTime;
    
    int fixedTimeMs = 1000;
    int fixedDepth = 5;
    int baseTimeMs = 300000;
    int incMs = 3000;
    
    char currentTurn = 'W';
    bool gameActive = false;
    bool engineThinking = false;
    bool isAnalyzing = false;

    void processEngineLine(const QString &line);
    void setupMenus();
    void setupKeybinds();
    void setupEditorUI(QVBoxLayout *layout);
    void startEngineMoveIfNeeded();
    void updateClockDisplays();
    void handleTimeOut(char loserSide);
};

#endif // MAINWINDOW_H