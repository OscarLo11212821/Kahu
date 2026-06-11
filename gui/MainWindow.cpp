#include "MainWindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QMenuBar>
#include <QShortcut>
#include <QPainter>
#include <QRegularExpression>

EvalBarWidget::EvalBarWidget(QWidget* parent) : QWidget(parent) {
    setFixedWidth(30);
}
void EvalBarWidget::setScore(int cp) {
    score = qBound(-2000, cp, 2000);
    update();
}
void EvalBarWidget::setNormMode(int mode) {
    normMode = mode;
    update();
}
void EvalBarWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.fillRect(rect(), Qt::black); 
    
    int h = height();
    double fillRatio = 0.5 + (score / 2000.0);
    fillRatio = qBound(0.0, fillRatio, 1.0);
    int whiteHeight = h * fillRatio;
    p.fillRect(0, h - whiteHeight, width(), whiteHeight, Qt::white);

    // Text formatting based on Normalization Mode
    QString txt;
    double val = score;
    if (normMode == 1) val /= 300.0;
    if (normMode == 2) val /= 100.0;

    if (normMode == 0) txt = QString("%1").arg(score > 0 ? "+" + QString::number(score) : QString::number(score));
    else txt = QString("%1").arg(val > 0 ? "+" + QString::number(val, 'f', 1) : QString::number(val, 'f', 1));

    p.setFont(QFont("Arial", 10, QFont::Bold));
    if (score >= 0) {
        p.setPen(Qt::black);
        p.drawText(QRect(0, h - 25, width(), 25), Qt::AlignBottom | Qt::AlignHCenter, txt);
    } else {
        p.setPen(Qt::white);
        p.drawText(QRect(0, 0, width(), 25), Qt::AlignTop | Qt::AlignHCenter, txt);
    }
}

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setupMenus();
    setupKeybinds();

    QWidget *central = new QWidget(this);
    QHBoxLayout *mainLayout = new QHBoxLayout(central);

    evalBar = new EvalBarWidget(this);
    mainLayout->addWidget(evalBar);

    boardWidget = new BoardWidget(this);
    mainLayout->addWidget(boardWidget, 2);

    QVBoxLayout *controlLayout = new QVBoxLayout();
    
    QGroupBox *statusGroup = new QGroupBox("Game Status");
    QVBoxLayout *statusGroupLayout = new QVBoxLayout(statusGroup);
    statusLabel = new QLabel("Turn: White");
    scoreLabel = new QLabel("Captured Red - W: 0 | B: 0");
    statusLabel->setStyleSheet("font-weight: bold; font-size: 14px;");
    statusGroupLayout->addWidget(statusLabel);
    statusGroupLayout->addWidget(scoreLabel);

    QHBoxLayout *evalComboLayout = new QHBoxLayout();
    comboEvalNorm = new QComboBox();
    comboEvalNorm->addItems({"Raw (cp)", "+1.0 per marble (÷300)", "+3.0 per marble (÷100)"});
    connect(comboEvalNorm, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int idx){ evalBar->setNormMode(idx); });
    evalComboLayout->addWidget(new QLabel("Eval View:"));
    evalComboLayout->addWidget(comboEvalNorm);
    statusGroupLayout->addLayout(evalComboLayout);
    
    controlLayout->addWidget(statusGroup);

    QGroupBox *clocksGroup = new QGroupBox("Clocks");
    QHBoxLayout *clocksLayout = new QHBoxLayout(clocksGroup);
    wClockLabel = new QLabel("05:00");
    bClockLabel = new QLabel("05:00");
    QString clockStyle = "font-family: monospace; font-size: 24px; font-weight: bold; background-color: black; color: %1; padding: 10px; border-radius: 5px;";
    wClockLabel->setStyleSheet(clockStyle.arg("white"));
    bClockLabel->setStyleSheet(clockStyle.arg("lightgray"));
    wClockLabel->setAlignment(Qt::AlignCenter);
    bClockLabel->setAlignment(Qt::AlignCenter);
    clocksLayout->addWidget(wClockLabel);
    clocksLayout->addWidget(bClockLabel);
    controlLayout->addWidget(clocksGroup);

    QHBoxLayout *btnLayout = new QHBoxLayout();
    QPushButton *btnNewGame = new QPushButton("New Game");
    QPushButton *btnSettings = new QPushButton("Settings");
    btnAnalysis = new QPushButton("Infinite Analysis");
    btnAnalysis->setCheckable(true);
    
    btnLayout->addWidget(btnNewGame);
    btnLayout->addWidget(btnSettings);
    btnLayout->addWidget(btnAnalysis);
    controlLayout->addLayout(btnLayout);

    setupEditorUI(controlLayout);

    logWindow = new QTextEdit();
    logWindow->setReadOnly(true);
    logWindow->setStyleSheet("background-color: #2b2b2b; color: #a9b7c6; font-family: monospace;");
    controlLayout->addWidget(logWindow);

    cmdInput = new QLineEdit();
    cmdInput->setPlaceholderText("Enter command (e.g., 'search millis 1000')");
    controlLayout->addWidget(cmdInput);

    mainLayout->addLayout(controlLayout, 1);
    setCentralWidget(central);

    gameTimer = new QTimer(this);
    connect(gameTimer, &QTimer::timeout, this, &MainWindow::onTimerTick);

    connect(cmdInput, &QLineEdit::returnPressed, [this]() { sendCommand(cmdInput->text()); cmdInput->clear(); });
    connect(btnNewGame, &QPushButton::clicked, this, &MainWindow::newGame);
    connect(btnSettings, &QPushButton::clicked, this, &MainWindow::openSettings);
    connect(btnAnalysis, &QPushButton::toggled, this, &MainWindow::toggleAnalysis);
    connect(boardWidget, &BoardWidget::moveRequested, this, &MainWindow::onUserMoveRequested);

    engineProcess = new QProcess(this);
    engineProcess->setProgram("./kuba_engine");
    connect(engineProcess, &QProcess::readyReadStandardOutput, this, &MainWindow::readEngineOutput);
    engineProcess->start();

    if(!engineProcess->waitForStarted()) {
        logWindow->append("<font color='red'>Failed to start engine!</font>");
    } else {
        sendCommand("hello");
        sendCommand("sync");
        sendCommand("getboard");
    }
}

MainWindow::~MainWindow() {
    if (engineProcess->state() == QProcess::Running) {
        sendCommand("quit");
        engineProcess->waitForFinished(2000);
    }
}

void MainWindow::setupEditorUI(QVBoxLayout *layout) {
    editorGroup = new QGroupBox("Board Editor");
    editorGroup->setCheckable(true);
    editorGroup->setChecked(false);
    
    QVBoxLayout *edLayout = new QVBoxLayout(editorGroup);

    QHBoxLayout *brushLayout = new QHBoxLayout();
    rbHand = new QRadioButton("Hand");
    rbWhite = new QRadioButton("W");
    rbBlack = new QRadioButton("B");
    rbRed = new QRadioButton("R");
    rbEmpty = new QRadioButton("Empty");
    rbHand->setChecked(true);
    brushLayout->addWidget(rbHand);
    brushLayout->addWidget(rbWhite);
    brushLayout->addWidget(rbBlack);
    brushLayout->addWidget(rbRed);
    brushLayout->addWidget(rbEmpty);
    edLayout->addLayout(brushLayout);

    QHBoxLayout *capLayout = new QHBoxLayout();
    spinCapW = new QSpinBox(); spinCapW->setRange(0, 7);
    spinCapB = new QSpinBox(); spinCapB->setRange(0, 7);
    capLayout->addWidget(new QLabel("W Cap:")); capLayout->addWidget(spinCapW);
    capLayout->addWidget(new QLabel("B Cap:")); capLayout->addWidget(spinCapB);
    edLayout->addLayout(capLayout);

    QHBoxLayout *btnLayout2 = new QHBoxLayout();
    QPushButton *btnInit = new QPushButton("Initial");
    QPushButton *btnClear = new QPushButton("Clear");
    btnLayout2->addWidget(btnInit);
    btnLayout2->addWidget(btnClear);
    edLayout->addLayout(btnLayout2);

    QHBoxLayout *kfenLayout = new QHBoxLayout();
    kfenInputEditor = new QLineEdit();
    QPushButton *btnApplyKfen = new QPushButton("Apply");
    kfenLayout->addWidget(new QLabel("KFEN:"));
    kfenLayout->addWidget(kfenInputEditor);
    kfenLayout->addWidget(btnApplyKfen);
    edLayout->addLayout(kfenLayout);

    layout->addWidget(editorGroup);
    
    connect(editorGroup, &QGroupBox::toggled, this, &MainWindow::toggleEditor);
    connect(rbHand, &QRadioButton::toggled, this, &MainWindow::onEditorBrushChanged);
    connect(rbWhite, &QRadioButton::toggled, this, &MainWindow::onEditorBrushChanged);
    connect(rbBlack, &QRadioButton::toggled, this, &MainWindow::onEditorBrushChanged);
    connect(rbRed, &QRadioButton::toggled, this, &MainWindow::onEditorBrushChanged);
    connect(rbEmpty, &QRadioButton::toggled, this, &MainWindow::onEditorBrushChanged);
    
    connect(spinCapW, QOverload<int>::of(&QSpinBox::valueChanged), [this](int){ boardWidget->setCaptured(spinCapW->value(), spinCapB->value()); onBoardEdited(); });
    connect(spinCapB, QOverload<int>::of(&QSpinBox::valueChanged), [this](int){ boardWidget->setCaptured(spinCapW->value(), spinCapB->value()); onBoardEdited(); });
    
    connect(btnInit, &QPushButton::clicked, [this](){ boardWidget->updateFromKFEN("WWWWWWW/W.....W/..RRR../.RRRRR./..RRR../B.....B/BBBBBBB w 0 0"); onBoardEdited(); });
    connect(btnClear, &QPushButton::clicked, [this](){ boardWidget->updateFromKFEN("7/7/7/7/7/7/7 w 0 0"); onBoardEdited(); });
    connect(btnApplyKfen, &QPushButton::clicked, [this](){ boardWidget->updateFromKFEN(kfenInputEditor->text()); onBoardEdited(); });

    connect(boardWidget, &BoardWidget::boardEdited, this, &MainWindow::onBoardEdited);
}

void MainWindow::setupMenus() {
    QMenu *gameMenu = menuBar()->addMenu("Game");
    gameMenu->addAction("New Game", QKeySequence("Ctrl+N"), this, &MainWindow::newGame);
    gameMenu->addAction("Settings...", QKeySequence("Ctrl+S"), this, &MainWindow::openSettings);
    gameMenu->addSeparator();
    gameMenu->addAction("Quit", QKeySequence("Ctrl+Q"), this, &QWidget::close);
}

void MainWindow::setupKeybinds() {
    new QShortcut(QKeySequence(Qt::Key_Space), this, SLOT(triggerEngineMove()));
}

void MainWindow::openSettings() {
    GameSettingsDialog dlg(this);
    if (dlg.exec() == QDialog::Accepted) {
        wPlayer = dlg.getWhitePlayer();
        bPlayer = dlg.getBlackPlayer();
        engineMode = dlg.getEngineMode();
        fixedTimeMs = dlg.getFixedTimeMs();
        fixedDepth = dlg.getFixedDepth();
        baseTimeMs = dlg.getBaseTimeMs();
        incMs = dlg.getIncrementMs();
        logWindow->append("<b>Settings updated. Start a New Game to apply.</b>");
    }
}

void MainWindow::readEngineOutput() {
    while (engineProcess->canReadLine()) {
        QString line = QString::fromStdString(engineProcess->readLine().toStdString()).trimmed();
        if (line.isEmpty()) continue;
        
        if (!line.startsWith("info")) {
            logWindow->append("Engine: " + line);
        }
        processEngineLine(line);
    }
}

void MainWindow::processEngineLine(const QString &line) {
    if (line.count('/') == 6) { // KFEN
        boardWidget->updateFromKFEN(line);
        QStringList parts = line.split(' ');
        if (parts.size() >= 4) {
            char newTurn = parts[1] == "w" ? 'W' : 'B';
            statusLabel->setText(newTurn == 'W' ? "Turn: White" : "Turn: Black");
            boardWidget->setSideToMove(newTurn);

            if (newTurn != currentTurn && gameActive) {
                if (engineMode == EngineMode::Tournament) {
                    if (currentTurn == 'W') wTimeMs += incMs; else bTimeMs += incMs;
                    updateClockDisplays();
                }
                currentTurn = newTurn;
                startEngineMoveIfNeeded();
            } else {
                currentTurn = newTurn;
            }
            
            int cW = parts[2].toInt();
            int cB = parts[3].toInt();
            scoreLabel->setText(QString("Captured Red - W: %1 | B: %2").arg(cW).arg(cB));
            
            spinCapW->blockSignals(true);
            spinCapB->blockSignals(true);
            spinCapW->setValue(cW);
            spinCapB->setValue(cB);
            spinCapW->blockSignals(false);
            spinCapB->blockSignals(false);
            
            if (editorGroup->isChecked()) {
                kfenInputEditor->setText(line);
            }
        }
    }
    else if (line.startsWith("info ")) {
        QRegularExpression scoreRe("score (cp )?(-?\\d+)");
        QRegularExpressionMatch m = scoreRe.match(line);
        if (m.hasMatch()) {
            int cp = m.captured(2).toInt();
            // Engine outputs STM evaluation. Convert to absolute (Global) evaluation
            if (currentTurn == 'B') cp = -cp; 
            evalBar->setScore(cp);
        }
        QRegularExpression pvRe("pv\\s+([a-gA-G1-7NnSsEeWw]+)");
        QRegularExpressionMatch pm = pvRe.match(line);
        if (pm.hasMatch()) {
            boardWidget->setPVMove(pm.captured(1));
        }
    }
    else if (line.startsWith("bestmove")) {
        engineThinking = false;
        QString move = line.section(' ', 1, 1);
        
        if (!isAnalyzing && gameActive) {
            sendCommand("move " + move);
            boardWidget->setLastMove(move);
            sendCommand("getboard");
            sendCommand("status");
        } else if (isAnalyzing) {
            boardWidget->setPVMove(move);
        }
    }
    else if (line == "white_wins" || line == "black_wins") {
        gameActive = false;
        gameTimer->stop();
        statusLabel->setText(line == "white_wins" ? "Game Over - White Wins!" : "Game Over - Black Wins!");
        statusLabel->setStyleSheet("color: red; font-weight: bold; font-size: 14px;");
    }
    else if (line.startsWith("score ")) {
        int cp = line.section(' ', 1, 1).toInt();
        // Convert static STM eval to absolute (Global)
        if (currentTurn == 'B') cp = -cp;
        evalBar->setScore(cp);
    }
}

void MainWindow::sendCommand(const QString &cmd) {
    if (engineProcess->state() != QProcess::Running) return;
    if (cmd != "getboard" && cmd != "status") { 
        logWindow->append("<b>User: " + cmd + "</b>");
    }
    engineProcess->write((cmd + "\n").toUtf8());
    if (cmd.startsWith("setboard ") || cmd == "reset" || cmd == "move ") {
        if (!cmd.startsWith("move")) sendCommand("getboard");
        if (isAnalyzing && !cmd.startsWith("search")) {
            sendCommand("search endless");
        }
    }
}

void MainWindow::onUserMoveRequested(const QString &move) {
    if (!gameActive && !isAnalyzing && !editorGroup->isChecked()) {
        sendCommand("move " + move);
        boardWidget->setLastMove(move);
        sendCommand("getboard");
        return;
    }
    if (engineThinking && !isAnalyzing) return;
    
    if (gameActive && ((currentTurn == 'W' && wPlayer == PlayerType::Engine) ||
        (currentTurn == 'B' && bPlayer == PlayerType::Engine))) {
        return;
    }

    if (isAnalyzing) sendCommand("halt");

    sendCommand("move " + move);
    boardWidget->setLastMove(move);
    sendCommand("getboard");
    sendCommand("status");
}

void MainWindow::newGame() {
    editorGroup->setChecked(false);
    btnAnalysis->setChecked(false);
    
    sendCommand("reset");
    gameActive = true;
    engineThinking = false;
    currentTurn = 'W';
    boardWidget->setLastMove("");
    boardWidget->setPVMove("");
    
    wTimeMs = (engineMode == EngineMode::Tournament) ? baseTimeMs : fixedTimeMs;
    bTimeMs = (engineMode == EngineMode::Tournament) ? baseTimeMs : fixedTimeMs;
    updateClockDisplays();
    gameTimer->start(100);

    statusLabel->setText("Turn: White");
    statusLabel->setStyleSheet("font-weight: bold; font-size: 14px; color: white;");
    QTimer::singleShot(200, this, &MainWindow::startEngineMoveIfNeeded); 
}

void MainWindow::startEngineMoveIfNeeded() {
    if (!gameActive || engineThinking || isAnalyzing) return;
    if ((currentTurn == 'W' && wPlayer == PlayerType::Engine) ||
        (currentTurn == 'B' && bPlayer == PlayerType::Engine)) {
        triggerEngineMove();
    } else {
        sendCommand("eval");
    }
}

void MainWindow::triggerEngineMove() {
    if (engineThinking) return;
    engineThinking = true;

    if (engineMode == EngineMode::FixedDepth) {
        sendCommand(QString("search depth %1").arg(fixedDepth));
    } else if (engineMode == EngineMode::FixedTime) {
        sendCommand(QString("search millis %1").arg(fixedTimeMs));
    } else {
        sendCommand(QString("search wtime %1 btime %2 wbonus %3 bbonus %4")
                    .arg(wTimeMs).arg(bTimeMs).arg(incMs).arg(incMs));
    }
}

void MainWindow::toggleAnalysis(bool active) {
    isAnalyzing = active;
    if (active) {
        gameActive = false;
        sendCommand("search endless");
    } else {
        sendCommand("halt");
        boardWidget->setPVMove("");
    }
}

void MainWindow::toggleEditor(bool active) {
    if (active) {
        if (isAnalyzing) btnAnalysis->setChecked(false);
        gameActive = false;
        boardWidget->setEditorMode(true);
        onEditorBrushChanged();
    } else {
        boardWidget->setEditorMode(false);
    }
}

void MainWindow::onEditorBrushChanged() {
    if (rbHand->isChecked()) boardWidget->setEditorBrush('M');
    else if (rbWhite->isChecked()) boardWidget->setEditorBrush('W');
    else if (rbBlack->isChecked()) boardWidget->setEditorBrush('B');
    else if (rbRed->isChecked()) boardWidget->setEditorBrush('R');
    else boardWidget->setEditorBrush('.');
}

void MainWindow::onBoardEdited() {
    QString kfen = boardWidget->getKFEN();
    sendCommand("setboard " + kfen);
    kfenInputEditor->setText(kfen);
}

void MainWindow::onTimerTick() {
    if (!gameActive || engineThinking || isAnalyzing) return;
    if (engineMode != EngineMode::Tournament) return;

    if (currentTurn == 'W') { wTimeMs -= 100; if (wTimeMs <= 0) handleTimeOut('W'); } 
    else { bTimeMs -= 100; if (bTimeMs <= 0) handleTimeOut('B'); }
    updateClockDisplays();
}

void MainWindow::updateClockDisplays() {
    auto fmt = [](int ms) {
        if (ms < 0) ms = 0;
        int s = ms / 1000;
        return QString("%1:%2").arg(s / 60, 2, 10, QChar('0')).arg(s % 60, 2, 10, QChar('0'));
    };
    wClockLabel->setText(fmt(wTimeMs));
    bClockLabel->setText(fmt(bTimeMs));

    QString act = "font-family: monospace; font-size: 24px; font-weight: bold; background-color: #225522; color: %1; padding: 10px; border-radius: 5px;";
    QString inact = "font-family: monospace; font-size: 24px; font-weight: bold; background-color: black; color: %1; padding: 10px; border-radius: 5px;";

    wClockLabel->setStyleSheet((currentTurn == 'W') ? act.arg("white") : inact.arg("white"));
    bClockLabel->setStyleSheet((currentTurn == 'B') ? act.arg("lightgray") : inact.arg("lightgray"));
}

void MainWindow::handleTimeOut(char loserSide) {
    gameActive = false;
    gameTimer->stop();
    QString msg = (loserSide == 'W') ? "Game Over - Black Wins on Time!" : "Game Over - White Wins on Time!";
    statusLabel->setText(msg);
    statusLabel->setStyleSheet("color: red; font-weight: bold; font-size: 14px;");
}