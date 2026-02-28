pipeline {
    agent any

    environment {
        IMAGE_NAME = "2022bcs0182malavika/2022bcs0182-malavika-wine-quality:v1"
        CONTAINER_NAME = "wine_test_container"
        NETWORK_NAME = "jenkins-net"
    }

    stages {

        stage('Pull Docker Image') {
            steps {
                echo "Pulling Docker image..."
                sh "docker pull ${IMAGE_NAME}"
            }
        }

        stage('Remove Old Container If Exists') {
            steps {
                echo "Removing old container if exists..."
                sh "docker rm -f ${CONTAINER_NAME} || true"
            }
        }

        stage('Run Container') {
            steps {
                echo "Starting container..."
                sh """
                docker run -d \
                --name ${CONTAINER_NAME} \
                --network ${NETWORK_NAME} \
                ${IMAGE_NAME}
                """
            }
        }

        stage('Wait for API to Start') {
            steps {
                sh """
                echo "Waiting for FastAPI service..."
                for i in {1..10}
                do
                    curl -f http://${CONTAINER_NAME}:8000/docs && exit 0
                    echo "Waiting..."
                    sleep 3
                done
                echo "API did not start in time"
                exit 1
                """
            }
        }

        stage('Valid Inference Test') {
            steps {
                sh """
                echo "Sending valid inference request..."

                RESPONSE=\$(curl -s -X POST http://${CONTAINER_NAME}:8000/predict \
                -H "Content-Type: application/json" \
                -d '{
                    "fixed_acidity": 7.4,
                    "volatile_acidity": 0.7,
                    "citric_acid": 0.0,
                    "residual_sugar": 1.9,
                    "chlorides": 0.076,
                    "free_sulfur_dioxide": 11.0,
                    "total_sulfur_dioxide": 34.0,
                    "density": 0.9978,
                    "pH": 3.51,
                    "sulphates": 0.56,
                    "alcohol": 9.4
                }')

                echo "Response: \$RESPONSE"

                echo \$RESPONSE | grep wine_quality
                echo \$RESPONSE | grep name
                echo \$RESPONSE | grep roll_no
                """
            }
        }

        stage('Invalid Inference Test') {
            steps {
                sh """
                echo "Sending invalid inference request..."

                STATUS=\$(curl -s -o /dev/null -w "%{http_code}" \
                -X POST http://${CONTAINER_NAME}:8000/predict \
                -H "Content-Type: application/json" \
                -d '{"fixed_acidity": 7.4}')

                echo "HTTP Status: \$STATUS"

                if [ "\$STATUS" -ne 422 ]; then
                    echo "Invalid request test failed"
                    exit 1
                fi
                """
            }
        }

        stage('Stop and Remove Container') {
            steps {
                echo "Stopping container..."
                sh "docker rm -f ${CONTAINER_NAME} || true"
            }
        }
    }

    post {
        always {
            echo "Final cleanup..."
            sh "docker rm -f ${CONTAINER_NAME} || true"
        }
    }
}
