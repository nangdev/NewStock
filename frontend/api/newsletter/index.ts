import { api } from 'api/axiosInstance';
import { API_PATH } from 'constants/api';
import { NewsletterReqType, NewsletterResType } from 'types/api/newsletter';

export const getNewsletterList = async ({ date }: NewsletterReqType) => {
  const response = await api.get<NewsletterResType>(`${API_PATH.NEWSLETTER}/${date}`, {});
  return response.data;
};
