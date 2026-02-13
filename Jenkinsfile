node {

    def DOCKER_IMAGE = "2022bcs0182malavika/ml-model"
    def CURRENT_ACCURACY = 0.0
    def IS_BETTER = false

    stage('Checkout') {
        checkout scm
    }

    stage('Setup Python Virtual Environment') {
        sh '''
        python3 -m venv venv
        . venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        '''
    }

    stage('Train Model') {
        sh '''
        . venv/bin/activate
        python train.py
        '''
    }

    stage('Read Accuracy') {
        def metrics = readJSON file: 'output/results.json'
        CURRENT_ACCURACY = metrics.accuracy as Double
        echo "Current Accuracy: ${CURRENT_ACCURACY}"
    }

    stage('Compare Accuracy') {
        def baseline = 0.0
        withCredentials([string(credentialsId: 'best-accuracy', variable: 'BEST_ACC')]) {
            baseline = BEST_ACC ? (BEST_ACC as Double) : 0.0
        }

        echo "Baseline Accuracy: ${baseline}"

        if (CURRENT_ACCURACY > baseline) {
            IS_BETTER = true
            echo "New model is better."
        } else {
            echo "New model is NOT better."
        }
    }

    if (IS_BETTER) {

        stage('Build Docker Image') {
            docker.build("${DOCKER_IMAGE}:${env.BUILD_NUMBER}")
        }

        stage('Push Docker Image') {
            docker.withRegistry('https://index.docker.io/v1/', 'dockerhub-creds') {
                docker.image("${DOCKER_IMAGE}:${env.BUILD_NUMBER}").push()
                docker.image("${DOCKER_IMAGE}:${env.BUILD_NUMBER}").push("latest")
            }
        }
    }

    stage('Archive Artifacts') {
        archiveArtifacts artifacts: 'output/**', fingerprint: true
    }
}
