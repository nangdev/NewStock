import { useQuery } from '@tanstack/react-query';
import { NewsletterReqType } from 'types/api/newsletter';

import { getNewsletterList } from '.';

export const useNewsletterListQuery = ({ date }: NewsletterReqType) => {
  return useQuery({
    queryKey: ['newsletter'],
    queryFn: () => getNewsletterList({ date }),
  });
};
