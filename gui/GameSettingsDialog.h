#ifndef GAMESETTINGSDIALOG_H
#define GAMESETTINGSDIALOG_H

#include <QDialog>
#include <QComboBox>
#include <QSpinBox>
#include <QStackedWidget>

enum class PlayerType { Human, Engine };
enum class EngineMode { FixedTime, FixedDepth, Tournament };

class GameSettingsDialog : public QDialog {
    Q_OBJECT

public:
    explicit GameSettingsDialog(QWidget *parent = nullptr);

    PlayerType getWhitePlayer() const;
    PlayerType getBlackPlayer() const;
    EngineMode getEngineMode() const;
    
    int getFixedTimeMs() const;
    int getFixedDepth() const;
    int getBaseTimeMs() const;
    int getIncrementMs() const;

private:
    QComboBox *comboWhite;
    QComboBox *comboBlack;
    QComboBox *comboMode;
    QStackedWidget *stackParams;
    
    QSpinBox *spinFixedTime;
    QSpinBox *spinFixedDepth;
    QSpinBox *spinBaseTimeMin;
    QSpinBox *spinIncrementSec;
};

#endif // GAMESETTINGSDIALOG_H